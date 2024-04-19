#include "bls12-381.cuh"
#include "commitment.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "tlookup.cuh"
#include "zkfc.cuh"
#include "zksoftmax.cuh"
#include "timer.hpp"

#include <iostream>
#include <cassert>

using namespace std;

int main(int argc, char **argv) {
    uint size = std::stoi(argv[1]);

    auto x = FrTensor::random(size);
    x.save("x.bin");
    FrTensor y("x.bin");

    cout << x.size << " " << y.size << endl;

    // check x == y
    auto u = random_vec(ceilLog2(size));
    cout << x(u) << endl;
    cout << y(u) << endl;

    auto com = Commitment::random(size);
    com.save("com.bin");
    Commitment com_("com.bin");
    cout << com.size << " " << com_.size << endl;
    cout << (com - com_)(u) << endl;

    return 0;
}