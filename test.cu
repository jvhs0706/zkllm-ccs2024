#include "bls12-381.cuh"
#include "commitment.cuh"
#include "fr-tensor.cuh"

#include <iostream>

using namespace std;

int main(int argc, char **argv) {
    uint nrow = stoi(argv[1]);
    uint ncol = stoi(argv[2]);

    auto generators = Commitment::random(ncol);

    // cout << generators(0) << endl;
    // cout << generators(generators.size - 1) << endl;

    FrTensor data = FrTensor::random(nrow * ncol);

    auto c = generators.commit(data, true);
    auto c_ = generators.commit(data, false);

    cout << c(0) << c_(0) << (c - c_)(0) << endl;
    cout << c(c.size >> 1) << c_(c.size >> 1) << (c - c_)(c.size >> 1) << endl;
    cout << c((c.size >> 1) + 1) << c_((c.size >> 1) + 1) << (c - c_)((c.size >> 1) + 1) << endl;
    cout << c(c.size - 2) << c_(c.size - 2) << (c - c_)(c.size - 2) << endl;
    cout << c(c.size - 1) << c_(c.size - 1) << (c - c_)(c.size - 1) << endl;

    return 0;
}