#include "bls12-381.cuh"
#include "commitment.cuh"
#include "fr-tensor.cuh"
#include "timer.hpp"

#include <iostream>
#include <cassert>

using namespace std;

int main(int argc, char **argv) {
    uint nrow = stoi(argv[1]);
    uint ncol = stoi(argv[2]);
    uint nbit = stoi(argv[3]);

    // make sure that nrow and ncol are both power of 2

    // assert((nrow & (nrow - 1)) == 0);
    // assert((ncol & (ncol - 1)) == 0);

    auto generators = Commitment::random(ncol);

    FrTensor data = FrTensor::random_int(nrow * ncol, nbit);

    Timer timer;
    timer.start();
    auto c = generators.commit(data);
    timer.stop();
    cout << "Commitment time: " << timer.getTotalTime() << endl;
    timer.reset();

    timer.start();
    auto c_ = generators.commit_int(data);
    timer.stop();
    cout << "Commitment time: " << timer.getTotalTime() << endl;
    timer.reset();

    auto u = random_vec(ceilLog2(nrow) + ceilLog2(ncol));
    std::vector<Fr_t> u0(u.end() - ceilLog2(nrow), u.end());
    std::vector<Fr_t> u1(u.begin(), u.end() - ceilLog2(nrow));

    cout << data.multi_dim_me({u0, u1}, {nrow, ncol}) << endl;
    timer.start();
    cout << generators.open(data, c, u) << endl;
    timer.stop();
    cout << "Open time: " << timer.getTotalTime() << endl;
    timer.reset();

    timer.start();
    cout << generators.open(data, c_, u) << endl;
    timer.stop();
    cout << "Open time: " << timer.getTotalTime() << endl;
    timer.reset();

    return 0;
}