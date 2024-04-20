#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include <string>

struct Weight {
    Commitment generator;
    FrTensor weight;
    G1TensorJacobian com;
};

Weight create_weight(string generator_filename, string weight_filename, string com_filename) {
    Commitment generator(generator_filename);
    FrTensor weight = FrTensor::from_int_bin(weight_filename);
    G1TensorJacobian com = generator.commit_int(weight);
    return {generator, weight, com};
}

int main(int argc, char *argv[])
{

    string input_file_name = argv[1];
    int seq_len = std::stoi(argv[2]);
    int embed_dim = std::stoi(argv[3]);
    string workdir = argv[4];
    string layer_prefix = argv[5];

    auto q_proj = create_weight(
        workdir + "/self_attn.q_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-commitment.bin"
    );

    auto k_proj = create_weight(
        workdir + "/self_attn.k_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-commitment.bin"
    );

    auto v_proj = create_weight(
        workdir + "/self_attn.v_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-commitment.bin"
    );

    auto o_proj = create_weight(
        workdir + "/self_attn.o_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-self_attn.o_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.o_proj.weight-commitment.bin"
    );

    zkFC q_layer(embed_dim, embed_dim, q_proj.weight);
    zkFC k_layer(embed_dim, embed_dim, k_proj.weight);
    zkFC v_layer(embed_dim, embed_dim, v_proj.weight);
    zkFC o_layer(embed_dim, embed_dim, o_proj.weight);

    FrTensor input = FrTensor::from_int_bin(input_file_name);
    cout << input.size << endl;
    auto Q = q_layer(input);
    auto K = k_layer(input);
    auto V = v_layer(input);

    cout << Q.size << " " << K.size << " " << V.size << endl;

    zkAttn attn(1L << 32, 1L << 32, {1 << 16, 1 << 16, 1 << 16, 1 << 16, 1 << 16}, 3, 0, {1.0 * (1L << 5), 1.0 * (1L << 11)}, seq_len, seq_len, embed_dim, 1 << 12);

    // CACHES
    FrTensor sm_in(seq_len * seq_len), sm_out(seq_len * seq_len), sm_shift(seq_len), sm_in_shifted(seq_len * seq_len);
    vector<FrTensor> sm_in_segments, sm_out_segments, sm_m_segments;

    cout << Q(0) << endl;
    cout << K(0) << endl;
    cout << V(0) << endl;


    auto Y = attn.compute(Q, K, V, sm_in, sm_out, sm_shift, sm_in_shifted, sm_in_segments, sm_out_segments, sm_m_segments);

    cout << Y(0) << endl;
    cout << Y(Y.size - 1) << endl;
    return 0;
}