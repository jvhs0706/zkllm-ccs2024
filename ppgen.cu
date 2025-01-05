#include "commitment.cuh"
#include <string>

int main(int argc, char *argv[]) {
    uint size = std::stoi(argv[1]);
    string filename = argv[2];

    Commitment commitment = Commitment::random(1 << ceilLog2(size));
    commitment.save(filename);
    return 0;
}