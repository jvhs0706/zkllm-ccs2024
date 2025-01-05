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
    int hidden_dim = std::stoi(argv[4]);
    string workdir = argv[5];
    string layer_prefix = argv[6];
    string output_file_name = argv[7];

    auto up_proj = create_weight(
        workdir + "/mlp.up_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-commitment.bin",
        embed_dim,
        hidden_dim
    );

    auto gate_proj = create_weight(
        workdir + "/mlp.gate_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-commitment.bin",
        embed_dim,
        hidden_dim
    );

    auto down_proj = create_weight(
        workdir + "/mlp.down_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.down_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.down_proj.weight-commitment.bin",
        hidden_dim,
        embed_dim
    );

    zkFC up_layer(embed_dim, hidden_dim, up_proj.weight);
    zkFC gate_layer(embed_dim, hidden_dim, gate_proj.weight);
    zkFC down_layer(hidden_dim, embed_dim, down_proj.weight);

    Rescaling up_rescale(1 << 16);
    Rescaling gate_rescale(1 << 20);
    Rescaling hidden_rescale(1 << 16);
    Rescaling down_rescale(1 << 16);

    FrTensor swiglu_values = FrTensor::from_int_bin("swiglu-table.bin");
    tLookupRangeMapping swiglu(-(1 << 21), 1 << 22, swiglu_values);

    FrTensor input = FrTensor::from_int_bin(input_file_name);
    auto up_out = up_layer(input);
    auto up_out_ = up_rescale(up_out);


    auto gate_out = gate_layer(input);
    auto gate_out_ = gate_rescale(gate_out);
    auto p = swiglu(gate_out_);

    auto &swiglu_out = p.first, &swiglu_m = p.second;

    auto temp_rand = random_vec(3);
    auto swiglu_u = random_vec(ceilLog2(seq_len * hidden_dim));
    auto swiglu_v = random_vec(ceilLog2(seq_len * hidden_dim));
    vector<Polynomial> swiglu_proof;
    

    auto down_in = swiglu_out * up_out_;
    auto down_in_ = hidden_rescale(down_in);



    auto down_out = down_layer(down_in_);
    auto down_out_ = down_rescale(down_out);

    down_out.save_int(output_file_name);

    down_rescale.prove(down_out, down_out_);
    verifyWeightClaim(down_proj, down_layer.prove(down_in_, down_out)[0]);

    hidden_rescale.prove(down_in, down_in_);
    swiglu.prove(gate_out_, swiglu_out, swiglu_m, temp_rand[0], temp_rand[1], temp_rand[2], swiglu_u, swiglu_v, swiglu_proof);
    cout << "SwiGLU proof complete." << endl;
    gate_rescale.prove(gate_out, gate_out_);
    verifyWeightClaim(gate_proj, gate_layer.prove(input, gate_out)[0]);

    up_rescale.prove(up_out, up_out_);
    verifyWeightClaim(up_proj, up_layer.prove(input, up_out)[0]);

    


    return 0;
}