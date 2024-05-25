#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include <string>

int main(int argc, char *argv[])
{
    string block_input_fn = argv[1];
    string block_output_fn = argv[2];
    string output_fn = argv[3];

    FrTensor x = FrTensor::from_int_bin(block_input_fn);
    FrTensor y = FrTensor::from_int_bin(block_output_fn);
    FrTensor z = x + y;
    z.save_int(output_fn);

    // cout << O_(0) << " " << O_(O_.size - 1) << endl;
    return 0;
}