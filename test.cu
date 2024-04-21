#include "bls12-381.cuh"
#include "commitment.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "tlookup.cuh"
#include "zkfc.cuh"
#include "zksoftmax.cuh"
#include "timer.hpp"
#include "rescaling.cuh"

#include <iostream>
#include <cassert>

using namespace std;

void check_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

int main(int argc, char **argv) {
    uint size = std::stoi(argv[1]);
    uint nbit = std::stoi(argv[2]);
    uint nbit_rescale = std::stoi(argv[3]);

    auto x = FrTensor::random_int(size, nbit);
    Rescaling rs(1 << nbit_rescale);
    // cout << rs.tl_rem.table << endl;

    auto x_ = rs(x);
    check_cuda_error(cudaGetLastError());
    // cout << x_ << endl;

    rs.prove(x, x_);

    return 0;
}