#ifndef TLOOKUP_CUH
#define TLOOKUP_CUH

#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 
#include "polynomial.cuh"
#include "proof.cuh"



class tLookup
{
    public:
    FrTensor table;
    tLookup(const FrTensor& table);
    
    // We do not directly use the values from the tensors. Instead, we assume that the tensors have been elementwisely converted to the indices of the table.
    FrTensor prep(const uint* indices, const uint D); // D - dimension of the tensor

    Fr_t prove(const FrTensor& S, const FrTensor& m, const Fr_t& alpha, const Fr_t& beta,
     const vector<Fr_t>& u, const vector<Fr_t>& v, vector<Polynomial>& proof);
};

class tLookupRange: public tLookup
{
    public:
    const int low;
    tLookupRange(int low, uint len);
    
    FrTensor prep(const int* vals, const uint D);
    FrTensor prep(const FrTensor& vals);
    
    using tLookup::prove;
};

class tLookupRangeMapping: public tLookupRange
{
    public:
    FrTensor mapped_vals;
    tLookupRangeMapping(int low, uint len, const FrTensor& mapped_vals);

    // direclty use prep and prove from tLookup
    
    using tLookupRange::prep;
    
    pair<FrTensor, FrTensor> operator()(const int* vals, const uint D);
    pair<FrTensor, FrTensor> operator()(const FrTensor& mvals);
    
    Fr_t prove(const FrTensor& S_in, const FrTensor& S_out, const FrTensor& m, 
        const Fr_t& r, const Fr_t& alpha, const Fr_t& beta,
        const vector<Fr_t>& u, const vector<Fr_t>& v, vector<Polynomial>& proof);
};

#endif