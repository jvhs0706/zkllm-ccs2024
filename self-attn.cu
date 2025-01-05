#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include <string>

int main(int argc, char *argv[])
{
    string mode = argv[1];
    string input_file_name = argv[2];
    uint seq_len = std::stoi(argv[3]);
    uint embed_dim = std::stoi(argv[4]);
    string workdir = argv[5];
    string layer_prefix = argv[6];
    string output_file_name = argv[7];

    if (mode == "linear")
    {
        auto q_proj = create_weight(
            workdir + "/self_attn.q_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-commitment.bin",
            embed_dim,
            embed_dim
        );

        auto k_proj = create_weight(
            workdir + "/self_attn.k_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-commitment.bin",
            embed_dim,
            embed_dim
        );

        auto v_proj = create_weight(
            workdir + "/self_attn.v_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-commitment.bin",
            embed_dim,
            embed_dim
        );
        zkFC q_layer(embed_dim, embed_dim, q_proj.weight);
        zkFC k_layer(embed_dim, embed_dim, k_proj.weight);
        zkFC v_layer(embed_dim, embed_dim, v_proj.weight);
        Rescaling q_rescale(1 << 16);
        Rescaling k_rescale(1 << 16);
        Rescaling v_rescale(1 << 16);

        FrTensor input = FrTensor::from_int_bin(input_file_name);
        auto Q = q_layer(input);
        auto Q_ = q_rescale(Q);

        auto K = k_layer(input);
        auto K_ = k_rescale(K);

        auto V = v_layer(input);
        auto V_ = v_rescale(V);
        
        q_rescale.prove(Q, Q_);
        k_rescale.prove(K, K_);
        v_rescale.prove(V, V_);

        verifyWeightClaim(k_proj, k_layer.prove(input, K)[0]);
        verifyWeightClaim(q_proj, q_layer.prove(input, Q)[0]);
        verifyWeightClaim(v_proj, v_layer.prove(input, V)[0]);

        Q_.save_int("temp_Q.bin");
        K_.save_int("temp_K.bin");
        V_.save_int("temp_V.bin");

        cout << "QKV linear proof successfully verified!" << endl;

        return 0;
    }

    else if (mode == "attn")
    {
        auto Q = FrTensor::from_int_bin("temp_Q.bin");
        auto K = FrTensor::from_int_bin("temp_K.bin");
        auto V = FrTensor::from_int_bin("temp_V.bin");
        auto d = Q.size / seq_len;
        
        auto X = FrTensor::matmul(Q, K.transpose(seq_len, d), seq_len, d, seq_len);

        zkSoftmax softmax({1<<8, 1<<20, 1<<20}, 1, 0, 1UL<<32, {1<<18, 1<<22}, seq_len, seq_len, d, 1);
        Rescaling rs1(1<< 20), rs2(1<<20);

        FrTensor shift(seq_len), X_shifted(seq_len * seq_len);
        vector<FrTensor> X_segments, Y_segments, m_segments;
        FrTensor Y = softmax.compute(X, shift, X_shifted, X_segments, Y_segments, m_segments);    
        Y.save_long("temp_head_Y.bin");
        
        
        auto out = FrTensor::matmul(Y, V, seq_len, seq_len, d);
        auto out_ = rs2(out);
        auto out__ = rs1(out_);

        out__.save_int("temp_head_out.bin");

        rs1.prove(out_, out__);
        rs2.prove(out, out_);
        auto temp_rand = random_vec(3);
        vector<Polynomial> proof;
        auto u1 = random_vec(ceilLog2(seq_len));
        auto u2 = random_vec(ceilLog2(d));
        auto ud = random_vec(ceilLog2(seq_len));
        auto claim = out.multi_dim_me({u1, u2}, {seq_len, d});
        auto final_claim = zkip(claim, Y.partial_me(u1, seq_len, seq_len), V.partial_me(u2, d, 1), ud, proof);

        softmax.prove(Y, X, shift, X_shifted, X_segments, Y_segments, m_segments, 
        random_vec(ceilLog2(Y.size)), random_vec(ceilLog2(Y.size)), temp_rand[0], temp_rand[1], temp_rand[2], proof);
        auto u1_ = random_vec(ceilLog2(seq_len));
        auto u2_ = random_vec(ceilLog2(seq_len));
        auto ud_ = random_vec(ceilLog2(d));
        auto claim_ = X.multi_dim_me({u1_, u2_}, {seq_len, seq_len});
        auto final_claim_ = zkip(claim_, Q.partial_me(u1_, seq_len, d), K.partial_me(u2_, seq_len, d), ud_, proof);
        cout << "Self attention proof successfully verified!" << endl; 
        return 0;
    }
    return 0;
}