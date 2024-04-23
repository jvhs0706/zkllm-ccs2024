#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include <string>

FrTensor computeHiddenIn(const FrTensor& up_out, const FrTensor& gate_out, vector<FrTensor>& cache);

void proveHiddenIn(const FrTensor& up_out, const FrTensor& gate_out, const FrTensor& hidden_in, vector<FrTensor>& cache);

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
    Rescaling gate_rescale(1 << 16);
    Rescaling hidden_rescale(1 << 16);
    Rescaling down_rescale(1 << 16);

    FrTensor input = FrTensor::from_int_bin(input_file_name);
    auto up_out = up_layer(input);
    auto up_out_ = up_rescale(up_out);


    auto gate_out = gate_layer(input);
    auto gate_out_ = gate_rescale(gate_out);

    vector<FrTensor> hidden_in_cache;
    auto down_in = computeHiddenIn(up_out_, gate_out_, hidden_in_cache);
    auto down_in_ = hidden_rescale(down_in);

    auto down_out = down_layer(down_in_);
    auto down_out_ = down_rescale(down_out);

    down_rescale.prove(down_out, down_out_);
    verifyWeightClaim(down_proj, down_layer.prove(down_in_, down_out)[0]);

    hidden_rescale.prove(down_in, down_in_);
    proveHiddenIn(up_out, gate_out, down_in, hidden_in_cache);
    
    gate_rescale.prove(gate_out, gate_out_);
    verifyWeightClaim(gate_proj, gate_layer.prove(input, gate_out)[0]);

    up_rescale.prove(up_out, up_out_);
    verifyWeightClaim(up_proj, up_layer.prove(input, up_out)[0]);


    return 0;
}

FrTensor computeHiddenIn(const FrTensor& up_out, const FrTensor& gate_out, vector<FrTensor>& cache)
{
    // specific for differnet LLMs; change this function to fit the model
    return up_out * gate_out;
}

void proveHiddenIn(const FrTensor& up_out, const FrTensor& gate_out, const FrTensor& hidden_in, vector<FrTensor>& cache)
{
    // specific for differnet LLMs; change this function to fit the model
}