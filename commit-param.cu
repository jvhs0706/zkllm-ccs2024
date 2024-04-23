#include "fr-tensor.cuh"
#include "commitment.cuh"

int main(int argc, char *argv[])
{
    string generator_filename = argv[1];
    string param_filename = argv[2];
    string output_filename = argv[3];
    uint in_dim = std::stoi(argv[4]);
    uint out_dim = std::stoi(argv[5]);

    Commitment generator(generator_filename);
    // generator.size has to be a power of 2
    if (generator.size != (1 << ceilLog2(generator.size))) throw std::runtime_error("Generator size has to be a power of 2");


    FrTensor param = FrTensor::from_int_bin(param_filename);
    cout << "Param size: " << param.size << endl;
    cout << "In dim: " << in_dim << endl;
    cout << "Out dim: " << out_dim << endl;
    auto param_padded = param.pad({in_dim, out_dim});
    auto com = generator.commit_int(param_padded);
    cout << "Generator size: " << generator.size << endl;
    cout << "Unpadded param size: " << param.size << endl;
    cout << "Padded param size: " << param_padded.size << endl;
    cout << "Commitment size: " << com.size << endl;

    com.save(output_filename);
    return 0;
}