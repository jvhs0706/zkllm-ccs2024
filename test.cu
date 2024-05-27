#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include <string>

int main(int argc, char *argv[])
{
    string Q_in = argv[1];
    string K_in = argv[2];
    string V_in = argv[3];

    uint m = std::stoi(argv[4]);
    uint n = std::stoi(argv[5]);
    uint d = std::stoi(argv[6]);
    string output_file_name = argv[7];

    FrTensor Q = FrTensor::from_int_bin(Q_in);
    FrTensor K = FrTensor::from_int_bin(K_in);
    auto X = FrTensor::matmul(Q, K.transpose(n, d), m, d, n);
    Rescaling rs1(1<< 20), rs2(1<<20);

    zkSoftmax softmax({1<<8, 1<<20, 1<<20}, 1, 0, 1UL<<32, {1<<18, 1<<22}, m, n, d, 1<<16);
    if (X.size != m * n)
    {
        std::cerr << "Input size mismatch" << std::endl;
        return 1;
    }

    FrTensor shift(m), X_shifted(m * n);
    vector<FrTensor> X_segments, Y_segments, m_segments;
    FrTensor Y = softmax.compute(X, shift, X_shifted, X_segments, Y_segments, m_segments);
    auto V = FrTensor::from_int_bin(V_in);
    auto out = FrTensor::matmul(Y, V, m, n, d);
    auto out_ = rs2(out);
    auto out__ = rs1(out_);

    out__.save_int(output_file_name);

    rs1.prove(out_, out__);
    rs2.prove(out, out_);
    auto temp_rand = random_vec(3);
    vector<Polynomial> proof;
    softmax.prove(Y, X, shift, X_shifted, X_segments, Y_segments, m_segments, 
    random_vec(ceilLog2(Y.size)), random_vec(ceilLog2(Y.size)), temp_rand[0], temp_rand[1], temp_rand[2], proof);

    return 0;
}