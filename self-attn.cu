#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include <string>

int main(int argc, char *argv[])
{
    string input_file_name = argv[1];
    int seq_len = std::stoi(argv[2]);
    int embed_dim = std::stoi(argv[3]);
    string workdir = argv[4];
    string layer_prefix = argv[5];
    string output_file_name = argv[6];

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

    Rescaling q_rescale(1 << 16);
    Rescaling k_rescale(1 << 16);
    Rescaling v_rescale(1 << 16);
    Rescaling o_rescale(1 << 16);
    Rescaling y_rescale(1 << 16);

    FrTensor input = FrTensor::from_int_bin(input_file_name);
    // cout << input.size << endl;
    auto Q = q_layer(input);
    auto Q_ = q_rescale(Q);
    q_rescale.prove(Q, Q_);
    auto K = k_layer(input);
    auto K_ = k_rescale(K);
    k_rescale.prove(K, K_);
    auto V = v_layer(input);
    auto V_ = v_rescale(V);
    v_rescale.prove(V, V_);

    zkAttn attn(1L << 16, 1L << 16, {1 << 16, 1 << 16, 1 << 16}, 1, 0, {1.0 * (1L << 5), 1.0 * (1L << 11)}, seq_len, seq_len, embed_dim, 1 << 12);

    // CACHES
    FrTensor sm_in(seq_len * seq_len), sm_out(seq_len * seq_len), sm_shift(seq_len), sm_in_shifted(seq_len * seq_len);
    vector<FrTensor> sm_in_segments, sm_out_segments, sm_m_segments;

    auto Y = attn.compute(Q_, K_, V_, sm_in, sm_out, sm_shift, sm_in_shifted, sm_in_segments, sm_out_segments, sm_m_segments);
    attn.prove(Q_, K_, V_, Y, sm_out, sm_in, sm_shift, sm_in_shifted, sm_in_segments, sm_out_segments, sm_m_segments);
    auto Y_ = y_rescale(Y);
    y_rescale.prove(Y, Y_);
    auto O = o_layer(Y_);
    auto O_ = o_rescale(O);
    o_rescale.prove(O, O_);

    O_.save_int(output_file_name);

    verifyWeightClaim(o_proj, o_layer.prove(Y_, O)[0]);
    verifyWeightClaim(k_proj, k_layer.prove(input, K)[0]);
    verifyWeightClaim(q_proj, q_layer.prove(input, Q)[0]);
    verifyWeightClaim(v_proj, v_layer.prove(input, V)[0]);

    // cout << O_(0) << " " << O_(O_.size - 1) << endl;
    return 0;
}