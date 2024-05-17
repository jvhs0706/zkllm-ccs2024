#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include <string>

FrTensor rotate_half(const FrTensor& X, uint seq_len, uint num_head, uint head_dim)
{
    auto X_T = X.transpose(seq_len * num_head, head_dim);
    auto x1 = X_T.trunc(0, X_T.size >> 1);
    auto x2 = X_T.trunc(X_T.size >> 1, X_T.size);
    return catTensors({-x2, x1}).transpose(head_dim, seq_len * num_head);
}

int main(int argc, char *argv[])
{
    string input_file_name = argv[1];
    uint seq_len = std::stoi(argv[2]);
    uint num_head = std::stoi(argv[3]);
    uint head_dim = std::stoi(argv[4]);
    uint embed_dim = num_head * head_dim;
    string workdir = argv[5];
    string layer_prefix = argv[6];
    string output_file_name = argv[7];

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

    auto o_proj = create_weight(
        workdir + "/self_attn.o_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-self_attn.o_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.o_proj.weight-commitment.bin",
        embed_dim,
        embed_dim
    );

    zkFC q_layer(embed_dim, embed_dim, q_proj.weight);
    zkFC k_layer(embed_dim, embed_dim, k_proj.weight);
    zkFC v_layer(embed_dim, embed_dim, v_proj.weight);
    zkFC o_layer(embed_dim, embed_dim, o_proj.weight);

    auto cos = FrTensor::from_int_bin("cos_temp.bin");
    auto sin = FrTensor::from_int_bin("sin_temp.bin");

    Rescaling q_rescale(1 << 16);
    Rescaling q_new_rescale(1 << 16);
    Rescaling k_rescale(1 << 16);
    Rescaling k_new_rescale(1 << 16);
    Rescaling v_rescale(1 << 16);
    Rescaling o_rescale(1 << 16);
    Rescaling y_rescale(1 << 16);

    FrTensor input = FrTensor::from_int_bin(input_file_name);
    // cout << input.size << endl;
    auto Q = q_layer(input);
    auto Q_ = q_rescale(Q);
    
    auto Q_rotated = rotate_half(Q_, seq_len, num_head, head_dim);
    auto Q_new = Q_ * cos + Q_rotated * sin;
    auto Q_new_ = q_new_rescale(Q_new);

    auto K = k_layer(input);
    auto K_ = k_rescale(K);
    
    auto K_rotated = rotate_half(K_, seq_len, num_head, head_dim);
    auto K_new = K_ * cos + K_rotated * sin;
    auto K_new_ = k_new_rescale(K_new);

    auto V = v_layer(input);
    auto V_ = v_rescale(V);
    

    zkAttn attn(1L << 16, 1L << 16, {1 << 16, 1 << 16, 1 << 16}, 1, 0, {1.0 * (1L << 5), 1.0 * (1L << 11)}, seq_len, seq_len, embed_dim, 1 << 12);

    // CACHES
    FrTensor sm_in(seq_len * seq_len), sm_out(seq_len * seq_len), sm_shift(seq_len), sm_in_shifted(seq_len * seq_len);
    vector<FrTensor> sm_in_segments, sm_out_segments, sm_m_segments;

    auto Y = attn.compute(Q_new_, K_new_, V_, sm_in, sm_out, sm_shift, sm_in_shifted, sm_in_segments, sm_out_segments, sm_m_segments);
    
    auto Y_ = y_rescale(Y);
    auto O = o_layer(Y_);
    auto O_ = o_rescale(O);
    

    O_.save_int(output_file_name);

    o_rescale.prove(O, O_);

    y_rescale.prove(Y, Y_);

    attn.prove(Q_new_, K_new_, V_, Y, sm_out, sm_in, sm_shift, sm_in_shifted, sm_in_segments, sm_out_segments, sm_m_segments);

    q_new_rescale.prove(Q_new, Q_new_);
    q_rescale.prove(Q, Q_);
    hadamard_product_sumcheck(Q_, cos, random_vec(ceilLog2(Q_.size)), random_vec(ceilLog2(Q_.size)));
    hadamard_product_sumcheck(Q_rotated, sin, random_vec(ceilLog2(Q_rotated.size)), random_vec(ceilLog2(Q_rotated.size)));
    k_new_rescale.prove(K_new, K_new_);
    k_rescale.prove(K, K_);
    hadamard_product_sumcheck(K_, cos, random_vec(ceilLog2(K_.size)), random_vec(ceilLog2(K_.size)));
    hadamard_product_sumcheck(K_rotated, sin, random_vec(ceilLog2(K_rotated.size)), random_vec(ceilLog2(K_rotated.size)));
    v_rescale.prove(V, V_);

    verifyWeightClaim(o_proj, o_layer.prove(Y_, O)[0]);
    verifyWeightClaim(k_proj, k_layer.prove(input, K)[0]);
    verifyWeightClaim(q_proj, q_layer.prove(input, Q)[0]);
    verifyWeightClaim(v_proj, v_layer.prove(input, V)[0]);

    // cout << O_(0) << " " << O_(O_.size - 1) << endl;
    return 0;
}