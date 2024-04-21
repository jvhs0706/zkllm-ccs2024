#ifndef RESCALING_CUH
#define RESCALING_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 
#include "tlookup.cuh"
#include "proof.cuh"

class Rescaling {
public:
    uint scaling_factor;
    tLookupRange tl_rem; // table for remainder
    Rescaling decomp(const FrTensor& X, FrTensor& rem);
    FrTensor *rem_tensor_ptr;

    Rescaling(uint scaling_factor);
    FrTensor operator()(const FrTensor& X);
    vector<Claim> prove(const FrTensor& X, const FrTensor& X_);
    ~Rescaling();
};

#endif // RESCALING_CUH