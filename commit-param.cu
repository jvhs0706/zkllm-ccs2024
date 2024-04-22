#include "fr-tensor.cuh"
#include "commitment.cuh"

int main(int argc, char *argv[])
{
    string generator_filename = argv[1];
    string param_filename = argv[2];
    string output_filename = argv[3];

    Commitment generator(generator_filename);
    FrTensor param = FrTensor::from_int_bin(param_filename);

    cout << "Generator size: " << generator.size << endl;
    cout << "Parameter size: " << param.size << endl;

    auto com = generator.commit_int(param);
    cout << "Commitment size: " << com.size << endl;
    com.save(output_filename);
    return 0;
}