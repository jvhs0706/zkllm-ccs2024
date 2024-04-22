#ifndef ZKFC_CUH
#define ZKFC_CUH

// #include <torch/torch.h>
// #include <torch/script.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "polynomial.cuh"



// KERNEL void matrixMultiplyOptimized(Fr_t* A, Fr_t* B, Fr_t* C, int rowsA, int colsA, int colsB);


// This is for computing
class zkFC {
public:
    const uint inputSize, outputSize;
    bool has_bias;
    FrTensor weights, bias;

    zkFC(uint input_size, uint output_size, const FrTensor& weight);
    zkFC(uint input_size, uint output_size, const FrTensor& weight, const FrTensor& bias);
    FrTensor operator()(const FrTensor& X) const;
    // void prove(const FrTensor& X, const FrTensor& Z, Commitment& generators) const;

    vector<Claim> prove(const FrTensor& X, const FrTensor& Y) const;

    static zkFC from_float_gpu_ptr (uint input_size, uint output_size, unsigned long scaling_factor, float* weight_ptr, float* bias_ptr);
    static zkFC from_float_gpu_ptr (uint input_size, uint output_size, unsigned long scaling_factor, float* weight_ptr);
    static FrTensor load_float_gpu_input(uint batch_size, uint input_size, unsigned long scaling_factor, float* input_ptr);

    // void attention(FrTensor &V, FrTensor &K, FrTensor &Q, FrTensor &out, uint rowsV, uint colsV, uint rowsK, uint colsK, uint rowsQ, uint colsQ);
};

// This is for proving
class zkFCStacked {
    public:
    bool has_bias;
    const uint num; 
    const uint batchSize;
    const uint inputSize;
    const uint outputSize;
    

    FrTensor X, W, b, Y; // num * batchSize * inputSize, num * inputSize * outputSize, num * outputSize, num * batchSize * outputSize

    zkFCStacked(bool has_bias, uint num, uint batch_size, uint input_size, uint output_size, const vector <zkFC>& layers, const vector <FrTensor>& Xs, const vector <FrTensor>& Ys);
    
    void prove(vector<Polynomial>& proof) const;

    void prove(const vector<Fr_t>& u_num, const vector<Fr_t>& v_num, const vector<Fr_t>& u_batch, const vector<Fr_t>& u_input, const vector<Fr_t>& u_output, vector<Polynomial>& proof) const;
};

// TODO: move this to somewhere else
// KERNEL void float_to_Fr_kernel(float* fs, Fr_t* frs, uint fs_num_window, uint frs_num_window, uint fs_window_size, uint frs_window_size);

Fr_t zkip(const Fr_t& claim, const FrTensor& a, const FrTensor& b, const vector<Fr_t>& u, vector<Polynomial>& proof);

Fr_t zkip_stacked(const Fr_t& claim, const FrTensor& A, const FrTensor& B, const vector<Fr_t>& uN, const vector<Fr_t>& uD, const vector<Fr_t> vN, uint N, uint D, vector<Polynomial>& proof);


#endif  // ZKFC_CUH
//