#include "proof.cuh"

void verifyWeightClaim(const Weight& w, const Claim& c)
{
    vector<Fr_t> u_cat = concatenate(vector<vector<Fr_t>>({c.u[1], c.u[0]}));
    auto w_padded = w.weight.pad({w.in_dim, w.out_dim});
    auto opening = w.generator.open(w_padded, w.com, u_cat);
    if (opening != c.claim) throw std::runtime_error("verifyWeightClaim: opening != c.claim");
    cout << "Opening complete" << endl;
}

KERNEL void Fr_ip_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;
    Fr_t a0 = (gid0 < in_size) ? a[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t b0 = (gid0 < in_size) ? b[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (gid1 < in_size) ? a[gid1] : blstrs__scalar__Scalar_ZERO;
    Fr_t b1 = (gid1 < in_size) ? b[gid1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = blstrs__scalar__Scalar_mul(a0, b0);
    out1[gid] = blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(a0, blstrs__scalar__Scalar_sub(b1, b0)), 
        blstrs__scalar__Scalar_mul(b0, blstrs__scalar__Scalar_sub(a1, a0)));
    out2[gid] = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_sub(a1, a0), blstrs__scalar__Scalar_sub(b1, b0));
}

void Fr_ip_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<Fr_t>& proof)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (begin >= end) {
        proof.push_back(a(0));
        proof.push_back(b(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    proof.push_back(out0.sum());
    proof.push_back(out1.sum());
    proof.push_back(out2.sum());

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(b.gpu_data, b_new.gpu_data, *begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_ip_sc(a_new, b_new, begin + 1, end, proof);
}

vector<Fr_t> inner_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u)
{
    vector<Fr_t> proof;
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (a.size <= (1 << (log_size))/2) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions");

    Fr_ip_sc(a, b, u.begin(), u.end(), proof);
    return proof;
}

void Fr_hp_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions 5");
    if (v_end - v_begin != u_end - u_begin) throw std::runtime_error("Incompatible dimensions 6");
    if (v_begin >= v_end) {
        proof.push_back(a(0));
        proof.push_back(b(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    vector<Fr_t> u_(u_begin + 1, u_end);
    //std::cout << u_.size() << "\t" << out0.size << "\t" << out1.size << "\t" << out2.size << std::endl;
    proof.push_back(out0(u_));
    proof.push_back(out1(u_));
    proof.push_back(out2(u_));

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(b.gpu_data, b_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_hp_sc(a_new, b_new, u_begin + 1, u_end, v_begin + 1, v_end, proof);
}

vector<Fr_t> hadamard_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u, vector<Fr_t> v)
{
    vector<Fr_t> proof;
    if (u.size() != v.size()) throw std::runtime_error("Incompatible dimensions 1");
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions 2");
    if (a.size <= (1 << (log_size - 1))) throw std::runtime_error("Incompatible dimensions 3");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions 4");

    Fr_hp_sc(a, b, u.begin(), u.end(), v.begin(), v.end(), proof);
    return proof;
}

KERNEL void Fr_bin_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    Fr_t a0 = (2 * gid < in_size) ? a[2 * gid] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (2 * gid + 1 < in_size) ? a[2 * gid + 1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_mul(a0, a0), a0);
    Fr_t diff = blstrs__scalar__Scalar_sub(a1, a0);
    out1[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_double(a0), diff), diff);
    out2[gid] = blstrs__scalar__Scalar_sqr(diff);
}

void Fr_bin_sc(const FrTensor& a, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof)
{
    if (v_end - v_begin != u_end - u_begin) throw std::runtime_error("Incompatible dimensions 6");
    if (v_begin >= v_end) {
        proof.push_back(a(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_bin_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    cudaDeviceSynchronize();
    vector<Fr_t> u_(u_begin + 1, u_end);
    //std::cout << u_.size() << "\t" << out0.size << "\t" << out1.size << "\t" << out2.size << std::endl;
    proof.push_back(out0(u_));
    proof.push_back(out1(u_));
    proof.push_back(out2(u_));

    FrTensor a_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *v_begin, in_size, out_size);
    cudaDeviceSynchronize();
    Fr_bin_sc(a_new, u_begin + 1, u_end, v_begin + 1, v_end, proof);
}

vector<Fr_t> binary_sumcheck(const FrTensor& a, vector<Fr_t> u, vector<Fr_t> v)
{
    vector<Fr_t> proof;
    if (u.size() != v.size()) throw std::runtime_error("Incompatible dimensions");
    uint log_size = u.size();
    if (a.size <= (1 << (log_size))/2) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions");

    Fr_bin_sc(a, u.begin(), u.end(), v.begin(), v.end(), proof);
    return proof;
}

// TODO: DEPRECATE ABOVE


bool operator==(const Fr_t& a, const Fr_t& b)
{
    return (a.val[0] == b.val[0] && a.val[1] == b.val[1] && a.val[2] == b.val[2] && a.val[3] == b.val[3] && a.val[4] == b.val[4] && a.val[5] == b.val[5] && a.val[6] == b.val[6] && a.val[7] == b.val[7]);
}

bool operator!=(const Fr_t& a, const Fr_t& b)
{
    return !(a == b);
}

KERNEL void hadamard_split_kernel(const Fr_t* in_ptr, Fr_t* out0, Fr_t* out1, uint N_out)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= N_out) return;
    out0[gid] = in_ptr[gid];
    out1[gid] = blstrs__scalar__Scalar_sub(in_ptr[gid + N_out], in_ptr[gid]);
}

KERNEL void hadamard_reduce_kernel(const Fr_t* in_ptr, Fr_t v, Fr_t* out, uint N_out)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= N_out) return;
    auto v_mont = blstrs__scalar__Scalar_mont(v);
    auto temp0 = in_ptr[gid];
    auto temp1 = blstrs__scalar__Scalar_sub(in_ptr[gid + N_out], in_ptr[gid]);
    out[gid] = blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(v_mont, temp1), temp0);
}

// Hadamard of multiple ones, Y and Xs should have been padded if their shapes are not a multiple of 2
Fr_t multi_hadamard_sumchecks(const Fr_t& claim, const vector<FrTensor>& Xs, const vector<Fr_t>& u, const vector<Fr_t>& v, vector<Polynomial>& proof)
{   
    auto N = Xs[0].size;
    //ensure N is a power of 2
    if (N == 1) return claim;
    if ((N & (N - 1)) != 0) throw std::runtime_error("N is not a power of 2");
    uint N_out = N >> 1;
    vector<FrTensor> out;

    FrTensor X0(N_out), X1(N_out);
    hadamard_split_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(Xs[0].gpu_data, X0.gpu_data, X1.gpu_data, N_out);
    cudaDeviceSynchronize();
    out.push_back(X0);
    out.push_back(X1);

    for (uint i = 1; i < Xs.size(); ++ i)
    {
        if (Xs[i].size != N) throw std::runtime_error("Xs[i] size is not N");
        FrTensor X0(N_out), X1(N_out);
        hadamard_split_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(Xs[i].gpu_data, X0.gpu_data, X1.gpu_data, N_out);
        cudaDeviceSynchronize();

        out.push_back(out.back() * X1);
        for (int j = i; j >= 1; -- j)
        {
            out[j] *= X0;
            out[j] += out[j - 1] * X1;
        }
        out[0] *= X0;
    }

    vector<Fr_t> coefs;
    vector<Fr_t> u_(u.begin(), u.end() - 1);
    for (auto& x : out) coefs.push_back(x(u_));
    proof.push_back(Polynomial(coefs));
    auto p = proof.back() * Polynomial::eq(u.back());

    if (claim != p({0, 0, 0, 0, 0, 0, 0, 0}) + p({1, 0, 0, 0, 0, 0, 0, 0})) throw std::runtime_error("multi_hadamard_sumchecks: claim != p(0) + p(1)");

    auto new_claim = proof.back()(v.back());
    vector<FrTensor> new_Xs;
    for (auto& X: Xs)
    {
        FrTensor new_X(N_out);
        hadamard_reduce_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(X.gpu_data, v.back(), new_X.gpu_data, N_out);
        cudaDeviceSynchronize();
        new_Xs.push_back(new_X);
    }

    return multi_hadamard_sumchecks(new_claim, new_Xs, u_, {v.begin(), v.end() - 1}, proof);
}