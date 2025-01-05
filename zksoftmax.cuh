#ifndef ZKSOFTMAX_CUH 
#define ZKSOFTMAX_CUH

#include "tlookup.cuh"
#include "zkfc.cuh"

class zkSoftmax {
    public:
    zkSoftmax(const vector<uint>& bs, uint L, uint M, unsigned long scaling_factor_in, const vector<double>& thetas, uint m, uint n, uint d, uint E);

    FrTensor compute(const FrTensor& X, FrTensor& shift, FrTensor& X_shifted, vector<FrTensor>& X_segments, vector<FrTensor>& Y_segments, vector<FrTensor>& m_segments);

    Fr_t prove(const FrTensor& Y, const FrTensor& X, const FrTensor& shift, const FrTensor& X_shifted,
        const vector<FrTensor>& X_segments, const vector<FrTensor>& Y_segments, const vector<FrTensor>& m_segments,
        const vector<Fr_t>& u_Y, const vector<Fr_t>& v_Y,
        const Fr_t& r_seg, const Fr_t& alpha_seg, const Fr_t& beta_seg, 
        vector<Polynomial>& proof);

    protected:
    const vector<uint> bs;
    // const vector<unsigned long> Bs; // length of each segment, and its cumulative prod
    const uint K, L, M; // the number of most and least significant segments
    const unsigned long scaling_factor_in; // the input of scaling factor (gamma**2)
    const vector<double> thetas; // the output scaling factor for each segment
    const uint m, n, d; // the dimensions of the input
    const uint E; // the error of the output
    
    vector<tLookupRange> least_significant_segments; // the lookup table for the least significant segments
    vector<tLookupRangeMapping> other_segments; // the lookup table for other segments
};

class zkAttn : public zkSoftmax {
    public:
    zkAttn(unsigned long sf_Q, unsigned long sf_K, const vector<uint>& bs, uint L, uint M, const vector<double>& thetas, uint m, uint n, uint d, uint E);

    // Q: m * d, K: n * d, V: n * d
    FrTensor compute(const FrTensor& Q, const FrTensor& K, const FrTensor& V, FrTensor& sm_in, FrTensor& sm_out,
    FrTensor& sm_shift, FrTensor& sm_in_shifted, vector<FrTensor>& sm_in_segments, vector<FrTensor>& sm_out_segments, vector<FrTensor>& sm_m_segments);

    Fr_t prove(const FrTensor& Q, const FrTensor& K, const FrTensor& V, const FrTensor& out,
        const FrTensor& sm_out, const FrTensor& sm_in, const FrTensor& sm_shift, const FrTensor& sm_in_shifted,
        const vector<FrTensor>& sm_in_segments, const vector<FrTensor>& sm_out_segments, const vector<FrTensor>& sm_m_segments,
        const vector<Fr_t>& u_matmul_out, const vector<Fr_t>& v_matmul_out, const vector<Fr_t>& w_matmul_out, 
        const vector<Fr_t>& v_sm, const Fr_t& r_seg, const Fr_t& alpha_seg, const Fr_t& beta_seg, 
        const vector<Fr_t>& v_matmul_in,
        vector<Polynomial>& proof);

    vector<Claim> prove(const FrTensor& Q, const FrTensor& K, const FrTensor& V, const FrTensor& out,
        const FrTensor& sm_out, const FrTensor& sm_in, const FrTensor& sm_shift, const FrTensor& sm_in_shifted,
        const vector<FrTensor>& sm_in_segments, const vector<FrTensor>& sm_out_segments, const vector<FrTensor>& sm_m_segments);
};

class zkAttnStacked : public zkAttn {
    public:
    zkAttnStacked(uint num, unsigned long sf_Q, unsigned long sf_K, const vector<uint>& bs, uint L, uint M, const vector<double>& thetas, uint m, uint n, uint d, uint E);

    Fr_t prove(const FrTensor& Q, const FrTensor& K, const FrTensor& V, const FrTensor& out,
        const FrTensor& sm_out, const FrTensor& sm_in, const FrTensor& sm_shift, const FrTensor& sm_in_shifted,
        const vector<FrTensor>& sm_in_segments, const vector<FrTensor>& sm_out_segments, const vector<FrTensor>& sm_m_segments,
        const vector<Fr_t>& u_matmul_out_num, const vector<Fr_t>& v_matmul_out_num,
        const vector<Fr_t>& u_matmul_out, const vector<Fr_t>& v_matmul_out, const vector<Fr_t>& w_matmul_out, 
        const vector<Fr_t>& v_sm, const Fr_t& r_seg, const Fr_t& alpha_seg, const Fr_t& beta_seg, 
        const vector<Fr_t>& v_matmul_in_num, const vector<Fr_t>& v_matmul_in,
        vector<Polynomial>& proof );

    vector<Claim> prove(const FrTensor& Q, const FrTensor& K, const FrTensor& V, const FrTensor& out,
        const FrTensor& sm_out, const FrTensor& sm_in, const FrTensor& sm_shift, const FrTensor& sm_in_shifted,
        const vector<FrTensor>& sm_in_segments, const vector<FrTensor>& sm_out_segments, const vector<FrTensor>& sm_m_segments);

    protected:
    const uint num;
};


#endif