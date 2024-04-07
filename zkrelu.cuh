#ifndef ZKRELU_CUH
#define ZKRELU_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 
#include "tlookup.cuh"
#include "proof.cuh"

class zkReLU {
public:
    uint scaling_factor;
    tLookupRange tl_rem; // table for remainder
    FrTensor decomp(const FrTensor& X, FrTensor& sign, FrTensor& abs, FrTensor& rem);
    FrTensor *sign_tensor_ptr, *abs_tensor_ptr, *rem_tensor_ptr, *m_tensor_ptr;

    zkReLU(uint scaling_factor);
    FrTensor operator()(const FrTensor& X);
    void prove(const FrTensor& Z, const FrTensor& A);
    ~zkReLU();
};



// DEVICE Fr_t ulong_to_scalar(unsigned long num);

// DEVICE unsigned long scalar_to_ulong(Fr_t num);

// KERNEL void relu_kernel(Fr_t* X, Fr_t* Z, Fr_t* sign, Fr_t* mag_bin, Fr_t* rem_bin, uint n);

#endif  // ZKRELU_CUH
