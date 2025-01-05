#include "zkfc.cuh"





zkFC::zkFC(uint input_size, uint output_size, const FrTensor& weight): inputSize(input_size), outputSize(output_size), has_bias(false), weights(weight), bias(output_size)
{
    if (weight.size != input_size * output_size) throw std::runtime_error("weight size does not match");
}

zkFC::zkFC(uint input_size, uint output_size, const FrTensor& weight, const FrTensor& bias): inputSize(input_size), outputSize(output_size), has_bias(true), weights(weight), bias(bias){
    if(weight.size != input_size * output_size) throw std::runtime_error("weight size does not match");
    if(bias.size != output_size) throw std::runtime_error("bias size does not match");
}

zkFC zkFC::from_float_gpu_ptr (uint input_size, uint output_size, unsigned long scaling_factor, float* weight_ptr, float* bias_ptr)
{
    FrTensor weights(input_size * output_size);
    FrTensor bias(output_size);
    float_to_scalar_kernel<<<(input_size * output_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(weight_ptr, weights.gpu_data, scaling_factor, input_size * output_size);
    cudaDeviceSynchronize();
    float_to_scalar_kernel<<<(output_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(bias_ptr, bias.gpu_data, scaling_factor * scaling_factor, output_size);
    cudaDeviceSynchronize();
    return zkFC(input_size, output_size, weights, bias);
}

zkFC zkFC::from_float_gpu_ptr (uint input_size, uint output_size, unsigned long scaling_factor, float* weight_ptr)
{
    FrTensor weights(input_size * output_size);
    float_to_scalar_kernel<<<(input_size * output_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(weight_ptr, weights.gpu_data, scaling_factor, input_size * output_size);
    cudaDeviceSynchronize();
    return zkFC(input_size, output_size, weights);
}

KERNEL void fcAddBiasKernel(Fr_t* Z, Fr_t* bias, uint numRow, uint numCol) // Z: numRow * numCol, bias: 1 * numCol
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRow * numCol) return;
    Z[tid] = blstrs__scalar__Scalar_add(Z[tid], bias[tid % numCol]);
}


FrTensor zkFC::operator()(const FrTensor& X) const { // X.size is batch_size * input_size

    if (X.size % inputSize) throw std::runtime_error("input size does not match");
    uint batchSize = X.size / inputSize;
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((outputSize + blockSize.x - 1) / blockSize.x, (batchSize + blockSize.y - 1) / blockSize.y);
    FrTensor out(batchSize * outputSize);
    matrixMultiplyOptimized<<<gridSize, blockSize>>>(X.gpu_data, weights.gpu_data, out.gpu_data, batchSize, inputSize, outputSize);
    cudaDeviceSynchronize();
    if(has_bias){
        fcAddBiasKernel<<<(batchSize * outputSize + FrNumThread - 1) / FrNumThread, FrNumThread>>>(out.gpu_data, bias.gpu_data, batchSize, outputSize);
        cudaDeviceSynchronize();
    }
    return out;
}


vector<Claim> zkFC::prove(const FrTensor& X, const FrTensor& Y) const
{
    if (has_bias) throw std::runtime_error("Cleaned-up version not implemented for zkFC with bias. Use zkFCStacked instead.");
    uint batchSize = X.size / inputSize;
    auto u_batch = random_vec(ceilLog2(batchSize));
    auto u_input = random_vec(ceilLog2(inputSize));
    auto u_output = random_vec(ceilLog2(outputSize));

    

    auto claim = Y.multi_dim_me({u_batch, u_output}, {batchSize, outputSize});

    auto X_reduced = X.partial_me(u_batch, batchSize, inputSize);
    auto W_reduced = weights.partial_me(u_output, outputSize, 1); // Y_reduced: num * inputSize
    vector<Polynomial> proof;
    auto final_claim = zkip(claim, X_reduced, W_reduced, u_input, proof);
    auto claim_X = X.multi_dim_me({u_batch, u_input}, {batchSize, inputSize});
    auto claim_W = weights.multi_dim_me({u_input, u_output}, {inputSize, outputSize});
    if (claim_X * claim_W != final_claim) {
        throw std::runtime_error("Claim does not match");
    }
    vector<Claim> claims;
    //Claim output_claim = {claim_W, &weights, vector<vector<Fr_t>>({u_input, u_output}), vector<uint>({inputSize, outputSize})};
    claims.push_back({claim_W, vector<vector<Fr_t>>({u_input, u_output}), vector<uint>({inputSize, outputSize})});
    return claims;

}


// zk inner product
const Fr_t TEMP_ZERO {0, 0, 0, 0, 0, 0, 0, 0};
const Fr_t TEMP_ONE {1, 0, 0, 0, 0, 0, 0, 0};

KERNEL void zkip_poly_kernel(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint N_in, uint N_out)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= N_out) return;
    
    uint gid0 = gid;
    uint gid1 = gid + N_out;
    Fr_t a0 = (gid0 < N_in) ? a[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t b0 = (gid0 < N_in) ? b[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (gid1 < N_in) ? a[gid1] : blstrs__scalar__Scalar_ZERO;
    Fr_t b1 = (gid1 < N_in) ? b[gid1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a0, b0));
    out1[gid] = blstrs__scalar__Scalar_mont(
        blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(a0, blstrs__scalar__Scalar_sub(b1, b0)), 
        blstrs__scalar__Scalar_mul(b0, blstrs__scalar__Scalar_sub(a1, a0)))
    );
    out2[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_sub(a1, a0), blstrs__scalar__Scalar_sub(b1, b0)));
}

KERNEL void zkip_reduce_kernel(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *new_a, GLOBAL Fr_t *new_b, Fr_t v, uint N_in, uint N_out)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= N_out) return;
    
    uint gid0 = gid;
    uint gid1 = gid + N_out;
    Fr_t v_mont = blstrs__scalar__Scalar_mont(v);
    Fr_t a0 = (gid0 < N_in) ? a[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t b0 = (gid0 < N_in) ? b[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (gid1 < N_in) ? a[gid1] : blstrs__scalar__Scalar_ZERO;
    Fr_t b1 = (gid1 < N_in) ? b[gid1] : blstrs__scalar__Scalar_ZERO;
    new_a[gid] = blstrs__scalar__Scalar_add(a0, blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(a1, a0)));
    new_b[gid] = blstrs__scalar__Scalar_add(b0, blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(b1, b0)));
}

Polynomial zkip_step_poly(const FrTensor& a, const FrTensor& b, const Fr_t& u)
{
    if (a.size != b.size) throw std::runtime_error("a.size != b.size");
    uint N_in = a.size, N_out = (1 << ceilLog2(a.size)) >> 1;
    FrTensor out0(N_out), out1(N_out), out2(N_out);
    zkip_poly_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, N_in, N_out);
    cudaDeviceSynchronize();
    return {{out0.sum(), out1.sum(), out2.sum()}};
    
}

Fr_t zkip(const Fr_t& claim, const FrTensor& a, const FrTensor& b, const vector<Fr_t>& u, vector<Polynomial>& proof)
{
    if (!u.size()) return claim;
    auto p = zkip_step_poly(a, b, u.back());
    proof.push_back(p);
    if (claim != p(TEMP_ZERO) + p(TEMP_ONE)) throw std::runtime_error("claim != p(0) + p(1)");
    uint N_in = a.size, N_out = (1 << ceilLog2(a.size)) >> 1;
    FrTensor new_a(N_out), new_b(N_out);
    zkip_reduce_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, new_a.gpu_data, new_b.gpu_data, u.back(), N_in, N_out);
    cudaDeviceSynchronize();
    return zkip(p(u.back()), new_a, new_b, {u.begin(), u.end()-1}, proof);
}

KERNEL void zkip_stacked_poly_kernel(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint N_in, uint N_out, uint D)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= N_out * D) return;
    
    Fr_t a0 = a[gid];
    Fr_t b0 = b[gid];
    Fr_t a1 = (gid + N_out * D < N_in * D) ? a[gid + N_out * D] : blstrs__scalar__Scalar_ZERO;
    a1 = blstrs__scalar__Scalar_sub(a1, a0);
    Fr_t b1 = (gid + N_out * D < N_in * D) ? b[gid + N_out * D] : blstrs__scalar__Scalar_ZERO;
    b1 = blstrs__scalar__Scalar_sub(b1, b0);
    out0[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a0, b0));
    out1[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(a0, b1), blstrs__scalar__Scalar_mul(a1, b0)));
    out2[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a1, b1));
}

KERNEL void zkip_stacked_reduce_kernel(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *new_a, GLOBAL Fr_t *new_b, Fr_t v, uint N_in, uint N_out, uint D)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= N_out * D) return;
    
    // uint gid0 = gid;
    // uint gid1 = gid + N_out * D;
    v = blstrs__scalar__Scalar_mont(v);
    Fr_t a0 = a[gid];
    Fr_t b0 = b[gid];
    Fr_t a1 = (gid + N_out * D < N_in * D) ? a[gid + N_out * D] : blstrs__scalar__Scalar_ZERO;
    a1 = blstrs__scalar__Scalar_sub(a1, a0);
    Fr_t b1 = (gid + N_out * D < N_in * D) ? b[gid + N_out * D] : blstrs__scalar__Scalar_ZERO;
    b1 = blstrs__scalar__Scalar_sub(b1, b0);
    new_a[gid] = blstrs__scalar__Scalar_add(a0, blstrs__scalar__Scalar_mul(v, a1));
    new_b[gid] = blstrs__scalar__Scalar_add(b0, blstrs__scalar__Scalar_mul(v, b1));
}

Polynomial zkip_stacked_step_poly(const FrTensor& A, const FrTensor& B, const vector<Fr_t>& u, uint N, uint D)
{   
    // cout << "Debug info: " << endl;
    // cout << "A.size = " << A.size << endl;
    // cout << "B.size = " << B.size << endl;
    // cout << "u.size = " << u.size() << endl;
    // cout << "N = " << N << endl;
    // cout << "D = " << D << endl;

    if (A.size != N * D) throw std::runtime_error("a.size != N * D");
    if (B.size != N * D) throw std::runtime_error("b.size != N * D");
    uint N_out = (1 << ceilLog2(N)) >> 1;
    // cout << "N_out = " << N_out << endl;
    uint size_out = N_out * D;
    // cout << "size_out = " << size_out << endl;
    FrTensor out0(size_out), out1(size_out), out2(size_out);
    zkip_stacked_poly_kernel<<<(size_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(A.gpu_data, B.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, N, N_out, D);
    cudaDeviceSynchronize();
    // cout << out0(size_out - 1) << endl;
    // cout << out1(size_out - 1) << endl;
    // cout << out2(size_out - 1) << endl;
    vector<Fr_t> u_ (u.begin(), u.end()-1);
    if (u_.size() > 0) return {{out0.partial_me(u_, N_out, D).sum(), out1.partial_me(u_, N_out, D).sum(), out2.partial_me(u_, N_out, D).sum()}};
    else return {{out0.sum(), out1.sum(), out2.sum()}};
    
}

Fr_t zkip_stacked(const Fr_t& claim, const FrTensor& A, const FrTensor& B, const vector<Fr_t>& uN, const vector<Fr_t>& uD, const vector<Fr_t> vN, uint N, uint D, vector<Polynomial>& proof)
{
    if (!uN.size()) return zkip(claim, A, B, uD, proof);
    auto p = zkip_stacked_step_poly(A, B, uN, N, D);
    proof.push_back(p);
    auto q = Polynomial::eq(uN.back()) * p;
    if (claim != q(TEMP_ZERO) + q(TEMP_ONE)) throw std::runtime_error("claim != q(0) + q(1)");
    uint N_out = (1 << ceilLog2(N)) >> 1;
    uint size_out = N_out * D;
    FrTensor new_A(size_out), new_B(size_out);
    zkip_stacked_reduce_kernel<<<(size_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(A.gpu_data, B.gpu_data, new_A.gpu_data, new_B.gpu_data, vN.back(), N, N_out, D);
    cudaDeviceSynchronize();
    // cout << new_A(size_out - 1) << endl;
    // cout << new_B(size_out - 1) << endl;
    return zkip_stacked(p(vN.back()), new_A, new_B, {uN.begin(), uN.end()-1}, uD, {vN.begin(), vN.end()-1}, N_out, D, proof);
}

// FrTensor catTensors(const vector<FrTensor>& vec){
//     // make sure all tensors have the same size
//     uint size = vec[0].size;
//     for (uint i = 1; i < vec.size(); i++){
//         if (vec[i].size != size) throw std::runtime_error("tensor size does not match");
//     }
//     uint num = vec.size();
//     FrTensor out(num * size);
    
//     // copy data to out
//     for (uint i = 0; i < num; i++){
//         cudaMemcpy(out.gpu_data + i * size, vec[i].gpu_data, size * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
//     }

//     return out;
// }

FrTensor catLayerWeights(const vector<zkFC>& layers){
    vector<FrTensor> weights;
    for (auto& layer: layers){
        weights.push_back(layer.weights);
    }
    return catTensors(weights);
}

FrTensor catLayerBiases(const vector<zkFC>& layers)
{
    vector<FrTensor> biases;
    for (auto& layer: layers){
        // if (!layer.has_bias) throw std::runtime_error("layer does not have bias");
        biases.push_back(layer.bias);
    }
    return catTensors(biases);
}

// Implmenetation of zkFCStacked follows
zkFCStacked::zkFCStacked(bool has_bias, uint num, uint batch_size, uint input_size, uint output_size, const vector <zkFC>& layers, const vector <FrTensor>& Xs, const vector <FrTensor>& Ys):
has_bias(has_bias), num(num), batchSize(batch_size), inputSize(input_size), outputSize(output_size),
X(catTensors(Xs)), Y(catTensors(Ys)), W(catLayerWeights(layers)), b(catLayerBiases(layers))
{   
    if (X.size != num * batchSize * inputSize) throw std::runtime_error("X size does not match");
    if (Y.size != num * batchSize * outputSize) throw std::runtime_error("Y size does not match");
    if (W.size != num * inputSize * outputSize) throw std::runtime_error("W size does not match");
    if (has_bias && b.size != num * outputSize) throw std::runtime_error("b size does not match");
}

void zkFCStacked::prove(vector<Polynomial>& proof) const
{
    auto u_num = random_vec(ceilLog2(num));
    auto v_num = random_vec(ceilLog2(num));
    auto u_batch = random_vec(ceilLog2(batchSize));
    auto u_input = random_vec(ceilLog2(inputSize));
    auto u_output = random_vec(ceilLog2(outputSize));

    prove(u_num, v_num, u_batch, u_input, u_output, proof);
}

void zkFCStacked::prove(const vector<Fr_t>& u_num, const vector<Fr_t>& v_num, const vector<Fr_t>& u_batch, const vector<Fr_t>& u_input, const vector<Fr_t>& u_output, vector<Polynomial>& proof) const
{
    auto claim = Y.multi_dim_me({u_num, u_batch, u_output}, {num, batchSize, outputSize});
    if (has_bias) 
    {   
        FrTensor broadcasting_ones(batchSize);
        //fill broadcasting_ones with 0s first
        cudaMemset(broadcasting_ones.gpu_data, 0, broadcasting_ones.size * sizeof(Fr_t));
        broadcasting_ones += {1, 0, 0, 0, 0, 0, 0, 0};

        claim = claim - broadcasting_ones(u_batch) * b.multi_dim_me({u_num, u_output}, {num, outputSize});
    }

    auto X_reduced = X.partial_me(u_batch, batchSize, inputSize);
    auto W_reduced = W.partial_me(u_output, outputSize, 1); // Y_reduced: num * inputSize

    auto final_claim = zkip_stacked(claim, X_reduced, W_reduced, u_num, u_input, v_num, num, inputSize, proof);
    auto opening = X.multi_dim_me({v_num, u_batch, u_input}, {num, batchSize, inputSize}) * W.multi_dim_me({v_num, u_input, u_output}, {num, inputSize, outputSize});
    if (final_claim != opening) throw std::runtime_error("final claim != opening");
}