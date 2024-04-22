#include "rescaling.cuh"

Rescaling::Rescaling(uint scaling_factor): scaling_factor(scaling_factor), tl_rem(-static_cast<int>(scaling_factor>>1), scaling_factor), rem_tensor_ptr(nullptr)
{
}

// void decomp(const FrTensor& X, FrTensor& sign, FrTensor& abs, FrTensor& rem, FrTensor& rem_ind);
KERNEL void rescaling_kernel(Fr_t* in_ptr, Fr_t* out_ptr, Fr_t* rem_ptr, long scaling_factor, uint N)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {   
        long hsf = scaling_factor >> 1;
        long x = scalar_to_long(in_ptr[tid]);
        long temp = (x + hsf) % scaling_factor;
        long x_rem = (temp < 0 ? temp + scaling_factor : temp) - hsf;
        long x_rescaled = (x - x_rem) / scaling_factor;
        out_ptr[tid] = long_to_scalar(x_rescaled);
        rem_ptr[tid] = long_to_scalar(x_rem);
    }
}

FrTensor Rescaling::operator()(const FrTensor& X)
{
    if (rem_tensor_ptr) delete rem_tensor_ptr;
    rem_tensor_ptr = new FrTensor(X.size);

    FrTensor out(X.size);
    uint block_size = 256;
    rescaling_kernel<<<(X.size + block_size - 1) / block_size, block_size>>>(X.gpu_data, out.gpu_data, rem_tensor_ptr->gpu_data, scaling_factor, X.size);
    cudaDeviceSynchronize();
    
    return out;
}

Rescaling::~Rescaling()
{
    if (rem_tensor_ptr) delete rem_tensor_ptr;
}

vector<Claim> Rescaling::prove(const FrTensor& X, const FrTensor& X_)
{
    if (X.size != X_.size)
    {
        throw std::runtime_error("Error: the size of X and X_ should be the same.");
    }

    auto u = random_vec(ceilLog2(X.size));
    auto v = random_vec(ceilLog2(X.size));
    
    auto rand_temp = random_vec(2);
    vector<Polynomial> proof;

    auto rem = rem_tensor_ptr -> pad({rem_tensor_ptr -> size});
    auto m = tl_rem.prep(rem);

    // cout << X << endl;
    // cout << X_ << endl;
    // cout << rem << endl;
    // cout << m << endl;
    // cout << tl_rem.table << endl;
    
    if (X(u) != X_(u) * Fr_t({scaling_factor, 0, 0, 0, 0, 0, 0, 0}) + rem(u))
    {
        throw std::runtime_error("Error: the rem is not correct.");
    }
    // cout << rem << endl;
    // cout << rem.sum() << endl;
    // cout << m << endl;
    // cout << tl_rem.table << endl;
    // cout << m*tl_rem.table << endl;
    // cout << (m*tl_rem.table).sum() << endl;
    tl_rem.prove(rem, m, rand_temp[0], rand_temp[1], u, v, proof);

    
    
    cout << "Rescaling proof complete." << endl;
    return {};
}