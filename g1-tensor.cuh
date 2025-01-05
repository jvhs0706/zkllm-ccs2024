#ifndef G1_TENSOR_CUH
#define G1_TENSOR_CUH

#include <iostream>
#include <iomanip>
#include "bls12-381.cuh"
#include "fr-tensor.cuh"
using namespace std;

typedef blstrs__fp__Fp Fp_t;
const uint G1NumThread = 64;
const uint G1AffineSharedMemorySize = 2 * sizeof(G1Affine_t) * G1NumThread; 
const uint G1JacobianSharedMemorySize = 2 * sizeof(G1Jacobian_t) * G1NumThread;

DEVICE Fp_t Fp_minus(Fp_t a);

DEVICE G1Affine_t G1Affine_minus(G1Affine_t a);

DEVICE G1Jacobian_t G1Jacobian_minus(G1Jacobian_t a);

ostream& operator<<(ostream& os, const Fp_t& x);

ostream& operator<<(ostream& os, const G1Affine_t& g);

ostream& operator<<(ostream& os, const G1Jacobian_t& g);


// x_mont = 0x120177419e0bfb75edce6ecc21dbf440f0ae6acdf3d0e747154f95c7143ba1c17817fc679976fff55cb38790fd530c16
const Fp_t G1_generator_x_mont = {
    4250078230,
    1555269520,
    2574712821,
    2014837863,
    339452353,
    357537223,
    4090554183,
    4037962445,
    568063040,
    3989728972,
    2651585397,
    302085953
};

// y_mont = 0xbbc3efc5008a26a0e1c8c3fad0059c051ac582950405194dd595f13570725ce8c22631a7918fd8ebaac93d50ce72271
const Fp_t G1_generator_y_mont = {
    216474225,
    3131872213,
    2031680910,
    2351063834,
    1460086222,
    3713621779,
    1346392468,
    1370249257,
    2902481344,
    236751935,
    1342743146,
    196886268
};

const Fp_t G1_ONE = {196605, 1980301312, 3289120770, 3958636555, 1405573306, 1598593111, 1884444485, 2010011731, 2723605613, 1543969431, 4202751123, 368467651};

const G1Affine_t G1Affine_generator {G1_generator_x_mont, G1_generator_y_mont};
const G1Jacobian_t G1Jacobian_generator {G1_generator_x_mont, G1_generator_y_mont, G1_ONE};

const G1Jacobian_t G1Jacobian_ZERO {};

class G1Tensor
{
    public:
    const uint size;

    G1Tensor(uint size);
};

class G1TensorAffine;
class G1TensorJacobian;

class G1TensorAffine: public G1Tensor
{
    public: 

    G1Affine_t* gpu_data;
    
    G1TensorAffine(const G1TensorAffine&);

    G1TensorAffine(uint size);

    G1TensorAffine(uint size, const G1Affine_t&);

    G1TensorAffine(uint size, const G1Affine_t* cpu_data);

    G1TensorAffine(const string& filename);

    ~G1TensorAffine();

    void save(const string& filename) const;

	G1Affine_t operator()(uint idx) const;
	// {
	// 	G1Affine_t out;
	// 	cudaMemcpy(&out, gpu_data + idx, sizeof(G1Affine_t), cudaMemcpyDeviceToHost);
	// 	return out;
	// }

    G1TensorAffine operator-() const;

    G1TensorJacobian& operator*(const FrTensor&);

    // friend class G1TensorJacobian;
};

const int G1RowwiseSumTileWidth = 16;
// class Commitment;

class G1TensorJacobian: public G1Tensor
{
    public: 
    
    G1Jacobian_t* gpu_data;

    G1TensorJacobian(const G1TensorJacobian&);

    G1TensorJacobian(uint size);

    G1TensorJacobian(uint size, const G1Jacobian_t&);

    G1TensorJacobian(uint size, const G1Jacobian_t* cpu_data);

    G1TensorJacobian(const G1TensorAffine& affine_tensor);

    G1TensorJacobian(const string& filename);

    ~G1TensorJacobian();

    void save(const string& filename) const;

	G1Jacobian_t operator()(uint) const;

    G1TensorJacobian operator-() const;

    G1TensorJacobian operator+(const G1TensorJacobian&) const;
    
    G1TensorJacobian operator+(const G1TensorAffine&) const;

    G1TensorJacobian operator+(const G1Jacobian_t&) const;

    G1TensorJacobian operator+(const G1Affine_t&) const;

    G1TensorJacobian& operator+=(const G1TensorJacobian&);
    
    G1TensorJacobian& operator+=(const G1TensorAffine&);

    G1TensorJacobian& operator+=(const G1Jacobian_t&);

    G1TensorJacobian& operator+=(const G1Affine_t&);

    G1TensorJacobian operator-(const G1TensorJacobian&) const;
    
    G1TensorJacobian operator-(const G1TensorAffine&) const;

    G1TensorJacobian operator-(const G1Jacobian_t&) const;

    G1TensorJacobian operator-(const G1Affine_t&) const;

    G1TensorJacobian& operator-=(const G1TensorJacobian&);
    
    G1TensorJacobian& operator-=(const G1TensorAffine&);

    G1TensorJacobian& operator-=(const G1Jacobian_t&);

    G1TensorJacobian& operator-=(const G1Affine_t&);

    G1Jacobian_t sum() const;

    G1TensorJacobian operator*(const FrTensor&) const;

    G1TensorJacobian& operator*=(const FrTensor&);

    G1Jacobian_t operator()(const vector<Fr_t>& u) const;

    G1TensorJacobian rowwise_sum(uint nrow, uint ncol) const;

    // friend G1Jacobian_t G1_me(const G1TensorJacobian& t, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end);

    // friend class G1TensorAffine;
    // friend class Commitment;
};

// Implement G1Affine

KERNEL void G1Affine_assign_broadcast(GLOBAL G1Affine_t* arr, GLOBAL G1Affine_t g, uint n);

KERNEL void G1_affine_elementwise_minus(GLOBAL G1Affine_t* arr_in, GLOBAL G1Affine_t* arr_out, uint n);


KERNEL void G1Jacobian_assign_broadcast(GLOBAL G1Jacobian_t* arr, G1Jacobian_t g, uint n);

KERNEL void G1_affine_to_jacobian(GLOBAL G1Affine_t* arr_affine, GLOBAL G1Jacobian_t* arr_jacobian, uint n);

KERNEL void G1_jacobian_elementwise_minus(GLOBAL G1Jacobian_t* arr_in, GLOBAL G1Jacobian_t* arr_out, uint n);


KERNEL void G1_jacobian_elementwise_add(GLOBAL G1Jacobian_t* arr1, GLOBAL G1Jacobian_t* arr2, GLOBAL G1Jacobian_t* arr_out, uint n);

KERNEL void G1_jacobian_broadcast_add(GLOBAL G1Jacobian_t* arr, G1Jacobian_t x, GLOBAL G1Jacobian_t* arr_out, uint n);

KERNEL void G1_jacobian_elementwise_madd(GLOBAL G1Jacobian_t* arr1, GLOBAL G1Affine_t* arr2, GLOBAL G1Jacobian_t* arr_out, uint n);

KERNEL void G1_jacobian_broadcast_madd(GLOBAL G1Jacobian_t* arr, G1Affine_t x, GLOBAL G1Jacobian_t* arr_out, uint n);


KERNEL void G1_jacobian_elementwise_sub(GLOBAL G1Jacobian_t* arr1, GLOBAL G1Jacobian_t* arr2, GLOBAL G1Jacobian_t* arr_out, uint n);

KERNEL void G1_jacobian_broadcast_sub(GLOBAL G1Jacobian_t* arr, G1Jacobian_t x, GLOBAL G1Jacobian_t* arr_out, uint n);

KERNEL void G1_jacobian_elementwise_msub(GLOBAL G1Jacobian_t* arr1, GLOBAL G1Affine_t* arr2, GLOBAL G1Jacobian_t* arr_out, uint n);

KERNEL void G1_jacobian_broadcast_msub(GLOBAL G1Jacobian_t* arr, G1Affine_t x, GLOBAL G1Jacobian_t* arr_out, uint n);


KERNEL void G1Jacobian_sum_reduction(GLOBAL G1Jacobian_t *arr, GLOBAL G1Jacobian_t *output, uint n);


DEVICE G1Jacobian_t G1Jacobian_mul(G1Jacobian_t a, Fr_t x);


KERNEL void G1_jacobian_elementwise_mul(GLOBAL G1Jacobian_t* arr_g1, GLOBAL Fr_t* arr_fr, GLOBAL G1Jacobian_t* arr_out, uint n);

KERNEL void G1_jacobian_elementwise_mul_broadcast(GLOBAL G1Jacobian_t* arr_g1, GLOBAL Fr_t* arr_fr, GLOBAL G1Jacobian_t* arr_out, uint n, uint m);


KERNEL void G1_me_step(GLOBAL G1Jacobian_t *arr_in, GLOBAL G1Jacobian_t *arr_out, Fr_t x, uint in_size, uint out_size);

// Given arr of shape of nrow * ncol_in, reduce to nrow * ncol_out, where ncol_out = (ncol + 1) >> 1
// KERNEL void G1Jacobian_rowwise_sum_step(const G1Jacobian_t* arr_in, G1Jacobian_t* arr_out, uint nrow, uint ncol_in, uint ncol_out);

KERNEL void G1Jacobian_rowwise_sum_reduction(const G1Jacobian_t* arr_in, G1Jacobian_t* arr_out, uint nrow, uint ncol, uint ncol_out);

#endif