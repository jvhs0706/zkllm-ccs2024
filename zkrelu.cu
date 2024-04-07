#include "zkrelu.cuh" 

zkReLU::zkReLU(uint scaling_factor): scaling_factor(scaling_factor), tl_rem(-static_cast<int>(scaling_factor>>1), scaling_factor), sign_tensor_ptr(nullptr), abs_tensor_ptr(nullptr), rem_tensor_ptr(nullptr), m_tensor_ptr(nullptr)
{
}

// void decomp(const FrTensor& X, FrTensor& sign, FrTensor& abs, FrTensor& rem, FrTensor& rem_ind);
KERNEL void zkrelu_decomp_kernel(Fr_t* X_ptr, Fr_t* sign_ptr, Fr_t* abs_ptr, Fr_t* rem_ptr, Fr_t* res_ptr, long scaling_factor, uint N)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {   
        long hsf = scaling_factor >> 1;
        long x = scalar_to_long(X_ptr[tid]);
        long temp = (x + hsf) % scaling_factor;
        long x_rem = temp < 0 ? temp + scaling_factor : temp;
        x_rem -= hsf;
        long x_rescaled = (x - x_rem) / scaling_factor;


        bool pos = x_rescaled >= 0;

        sign_ptr[tid] = {static_cast<uint>(pos), 0, 0, 0, 0, 0, 0, 0};
        abs_ptr[tid] = pos? long_to_scalar(x_rescaled) : long_to_scalar(-x_rescaled);
        rem_ptr[tid] = long_to_scalar(x_rem);
        res_ptr[tid] = pos? long_to_scalar(x_rescaled) : blstrs__scalar__Scalar_ZERO;
    }
}

FrTensor zkReLU::decomp(const FrTensor& X, FrTensor& sign, FrTensor& abs, FrTensor& rem)
{
    uint N = X.size;
    FrTensor res(N);
    uint block_size = 256;
    uint grid_size = (N + block_size - 1) / block_size;
    zkrelu_decomp_kernel<<<grid_size, block_size>>>(X.gpu_data, sign.gpu_data, abs.gpu_data, rem.gpu_data, res.gpu_data, static_cast<long>(scaling_factor), N);
    cudaDeviceSynchronize();
    return res;
}

FrTensor zkReLU::operator()(const FrTensor& X)
{
    if (sign_tensor_ptr) delete sign_tensor_ptr;
    sign_tensor_ptr = new FrTensor(X.size);
    if (abs_tensor_ptr) delete abs_tensor_ptr;
    abs_tensor_ptr = new FrTensor(X.size);
    if (rem_tensor_ptr) delete rem_tensor_ptr;
    rem_tensor_ptr = new FrTensor(X.size);
    if (m_tensor_ptr) delete m_tensor_ptr;
    // m_tensor_ptr = new FrTensor(tl_rem.table.size);

    FrTensor res = decomp(X, *sign_tensor_ptr, *abs_tensor_ptr, *rem_tensor_ptr);
    m_tensor_ptr = new FrTensor(tl_rem.prep(*rem_tensor_ptr));
    return res;
}

void zkReLU::prove(const FrTensor& Z, const FrTensor& A)
{
}

zkReLU::~zkReLU()
{
    if (sign_tensor_ptr) delete sign_tensor_ptr;
    if (abs_tensor_ptr) delete abs_tensor_ptr;
    if (rem_tensor_ptr) delete rem_tensor_ptr;
    if (m_tensor_ptr) delete m_tensor_ptr;
}