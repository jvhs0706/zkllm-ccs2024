#ifndef BLS12_381_CUH
#define BLS12_381_CUH

// Defines to make the code work with both, CUDA and OpenCL
#ifdef __NVCC__
  #define DEVICE __device__
  #define GLOBAL
  #define KERNEL extern "C" __global__
  #define LOCAL __shared__
  #define CONSTANT __constant__

  #define GET_GLOBAL_ID() blockIdx.x * blockDim.x + threadIdx.x
  #define GET_GROUP_ID() blockIdx.x
  #define GET_LOCAL_ID() threadIdx.x
  #define GET_LOCAL_SIZE() blockDim.x
  #define BARRIER_LOCAL() __syncthreads()

  typedef unsigned char uchar;

  #define BLS12_381_CUH_CUDA
#else // OpenCL
  #define DEVICE
  #define GLOBAL __global
  #define KERNEL __kernel
  #define LOCAL __local
  #define CONSTANT __constant

  #define GET_GLOBAL_ID() get_global_id(0)
  #define GET_GROUP_ID() get_group_id(0)
  #define GET_LOCAL_ID() get_local_id(0)
  #define GET_LOCAL_SIZE() get_local_size(0)
  #define BARRIER_LOCAL() barrier(CLK_LOCAL_MEM_FENCE)
#endif

#ifdef __NV_CL_C_VERSION
#define BLS12_381_CUH_OPENCL_NVIDIA
#endif

#if defined(__WinterPark__) || defined(__BeaverCreek__) || defined(__Turks__) || \
    defined(__Caicos__) || defined(__Tahiti__) || defined(__Pitcairn__) || \
    defined(__Capeverde__) || defined(__Cayman__) || defined(__Barts__) || \
    defined(__Cypress__) || defined(__Juniper__) || defined(__Redwood__) || \
    defined(__Cedar__) || defined(__ATI_RV770__) || defined(__ATI_RV730__) || \
    defined(__ATI_RV710__) || defined(__Loveland__) || defined(__GPU__) || \
    defined(__Hawaii__)
#define AMD
#endif

// Returns a * b + c + d, puts the carry in d
DEVICE ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d);

// Returns a + b, puts the carry in d
DEVICE ulong add_with_carry_64(ulong a, ulong *b);

// Returns a * b + c + d, puts the carry in d
DEVICE uint mac_with_carry_32(uint a, uint b, uint c, uint *d);

// Returns a + b, puts the carry in b
DEVICE uint add_with_carry_32(uint a, uint *b);

// Reverse the given bits. It's used by the FFT kernel.
DEVICE uint bitreverse(uint n, uint bits);

#ifdef BLS12_381_CUH_CUDA
// CUDA doesn't support local buffers ("dynamic shared memory" in CUDA lingo) as function
// arguments, but only a single globally defined extern value. Use `uchar` so that it is always
// allocated by the number of bytes.
extern LOCAL uchar cuda_shared[];

typedef uint uint32_t;
typedef int  int32_t;
typedef uint limb;

DEVICE inline uint32_t add_cc(uint32_t a, uint32_t b);

DEVICE inline uint32_t addc_cc(uint32_t a, uint32_t b);

DEVICE inline uint32_t addc(uint32_t a, uint32_t b);



DEVICE inline uint32_t madlo(uint32_t a, uint32_t b, uint32_t c);

DEVICE inline uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c);

DEVICE inline uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c);

DEVICE inline uint32_t madloc(uint32_t a, uint32_t b, uint32_t c) ;

DEVICE inline uint32_t madhi(uint32_t a, uint32_t b, uint32_t c) ;

DEVICE inline uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) ;

DEVICE inline uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c) ;

DEVICE inline uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) ;

typedef struct {
  int32_t _position;
} chain_t;

DEVICE inline
void chain_init(chain_t *c) ;

DEVICE inline
uint32_t chain_add(chain_t *ch, uint32_t a, uint32_t b) ;

DEVICE inline
uint32_t chain_madlo(chain_t *ch, uint32_t a, uint32_t b, uint32_t c);

DEVICE inline
uint32_t chain_madhi(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) ;

#endif

#define blstrs__scalar__Scalar_limb uint
#define blstrs__scalar__Scalar_LIMBS 8
#define blstrs__scalar__Scalar_LIMB_BITS 32
#define blstrs__scalar__Scalar_INV 4294967295
typedef struct { blstrs__scalar__Scalar_limb val[blstrs__scalar__Scalar_LIMBS]; } blstrs__scalar__Scalar;
extern CONSTANT blstrs__scalar__Scalar blstrs__scalar__Scalar_ONE;
extern CONSTANT blstrs__scalar__Scalar blstrs__scalar__Scalar_P;
extern CONSTANT blstrs__scalar__Scalar blstrs__scalar__Scalar_R2;
extern CONSTANT blstrs__scalar__Scalar blstrs__scalar__Scalar_ZERO;
#if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)

DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_sub_nvidia(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b);
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_add_nvidia(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) ;
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define blstrs__scalar__Scalar_BITS (blstrs__scalar__Scalar_LIMBS * blstrs__scalar__Scalar_LIMB_BITS)
#if blstrs__scalar__Scalar_LIMB_BITS == 32
  #define blstrs__scalar__Scalar_mac_with_carry mac_with_carry_32
  #define blstrs__scalar__Scalar_add_with_carry add_with_carry_32
#elif blstrs__scalar__Scalar_LIMB_BITS == 64
  #define blstrs__scalar__Scalar_mac_with_carry mac_with_carry_64
  #define blstrs__scalar__Scalar_add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool blstrs__scalar__Scalar_gte(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b);

// Equals
DEVICE bool blstrs__scalar__Scalar_eq(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b);

// Normal addition
#if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)
  #define blstrs__scalar__Scalar_add_ blstrs__scalar__Scalar_add_nvidia
  #define blstrs__scalar__Scalar_sub_ blstrs__scalar__Scalar_sub_nvidia
#else
  DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_add_(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b);
  blstrs__scalar__Scalar blstrs__scalar__Scalar_sub_(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b);
#endif

// Modular subtraction
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b);

// Modular addition
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_add(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b);

#ifdef BLS12_381_CUH_CUDA
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void blstrs__scalar__Scalar_reduce(uint32_t accLow[blstrs__scalar__Scalar_LIMBS], uint32_t np0, uint32_t fq[blstrs__scalar__Scalar_LIMBS]);

// Requirement: yLimbs >= xLimbs
DEVICE inline
void blstrs__scalar__Scalar_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy);

DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_mul_nvidia(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b);

#endif

// Modular multiplication
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_mul_default(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b);

#ifdef BLS12_381_CUH_CUDA
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b);
#else
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b);
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_sqr(blstrs__scalar__Scalar a);

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of blstrs__scalar__Scalar_add(a, a)
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_double(blstrs__scalar__Scalar a);
// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_pow(blstrs__scalar__Scalar base, uint exponent);


// Store squares of the base in a lookup table for faster evaluation.
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_pow_lookup(GLOBAL blstrs__scalar__Scalar *bases, uint exponent);
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar a);
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_unmont(blstrs__scalar__Scalar a);

DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_inverse(blstrs__scalar__Scalar a);
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_div(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b);

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool blstrs__scalar__Scalar_get_bit(blstrs__scalar__Scalar l, uint i);

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint blstrs__scalar__Scalar_get_bits(blstrs__scalar__Scalar l, uint skip, uint window);
#define blstrs__fp__Fp_limb uint
#define blstrs__fp__Fp_LIMBS 12
#define blstrs__fp__Fp_LIMB_BITS 32
#define blstrs__fp__Fp_INV 4294770685
typedef struct { blstrs__fp__Fp_limb val[blstrs__fp__Fp_LIMBS]; } blstrs__fp__Fp;
extern CONSTANT blstrs__fp__Fp blstrs__fp__Fp_ONE;
extern CONSTANT blstrs__fp__Fp blstrs__fp__Fp_P;
extern CONSTANT blstrs__fp__Fp blstrs__fp__Fp_R2;
extern CONSTANT blstrs__fp__Fp blstrs__fp__Fp_ZERO;
#if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)

DEVICE blstrs__fp__Fp blstrs__fp__Fp_sub_nvidia(blstrs__fp__Fp a, blstrs__fp__Fp b);
DEVICE blstrs__fp__Fp blstrs__fp__Fp_add_nvidia(blstrs__fp__Fp a, blstrs__fp__Fp b);
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define blstrs__fp__Fp_BITS (blstrs__fp__Fp_LIMBS * blstrs__fp__Fp_LIMB_BITS)
#if blstrs__fp__Fp_LIMB_BITS == 32
  #define blstrs__fp__Fp_mac_with_carry mac_with_carry_32
  #define blstrs__fp__Fp_add_with_carry add_with_carry_32
#elif blstrs__fp__Fp_LIMB_BITS == 64
  #define blstrs__fp__Fp_mac_with_carry mac_with_carry_64
  #define blstrs__fp__Fp_add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool blstrs__fp__Fp_gte(blstrs__fp__Fp a, blstrs__fp__Fp b);

// Equals
DEVICE bool blstrs__fp__Fp_eq(blstrs__fp__Fp a, blstrs__fp__Fp b);

// Normal addition
#if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)
  #define blstrs__fp__Fp_add_ blstrs__fp__Fp_add_nvidia
  #define blstrs__fp__Fp_sub_ blstrs__fp__Fp_sub_nvidia
#else
  DEVICE blstrs__fp__Fp blstrs__fp__Fp_add_(blstrs__fp__Fp a, blstrs__fp__Fp b);
  blstrs__fp__Fp blstrs__fp__Fp_sub_(blstrs__fp__Fp a, blstrs__fp__Fp b);
#endif

// Modular subtraction
DEVICE blstrs__fp__Fp blstrs__fp__Fp_sub(blstrs__fp__Fp a, blstrs__fp__Fp b);

// Modular addition
DEVICE blstrs__fp__Fp blstrs__fp__Fp_add(blstrs__fp__Fp a, blstrs__fp__Fp b);


#ifdef BLS12_381_CUH_CUDA
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void blstrs__fp__Fp_reduce(uint32_t accLow[blstrs__fp__Fp_LIMBS], uint32_t np0, uint32_t fq[blstrs__fp__Fp_LIMBS]);

// Requirement: yLimbs >= xLimbs
DEVICE inline
void blstrs__fp__Fp_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy);

DEVICE blstrs__fp__Fp blstrs__fp__Fp_mul_nvidia(blstrs__fp__Fp a, blstrs__fp__Fp b);

#endif

// Modular multiplication
DEVICE blstrs__fp__Fp blstrs__fp__Fp_mul_default(blstrs__fp__Fp a, blstrs__fp__Fp b);

#ifdef BLS12_381_CUH_CUDA
DEVICE blstrs__fp__Fp blstrs__fp__Fp_mul(blstrs__fp__Fp a, blstrs__fp__Fp b);
#else
DEVICE blstrs__fp__Fp blstrs__fp__Fp_mul(blstrs__fp__Fp a, blstrs__fp__Fp b);
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE blstrs__fp__Fp blstrs__fp__Fp_sqr(blstrs__fp__Fp a);

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of blstrs__fp__Fp_add(a, a)
DEVICE blstrs__fp__Fp blstrs__fp__Fp_double(blstrs__fp__Fp a);

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE blstrs__fp__Fp blstrs__fp__Fp_pow(blstrs__fp__Fp base, uint exponent);

// Store squares of the base in a lookup table for faster evaluation.
DEVICE blstrs__fp__Fp blstrs__fp__Fp_pow_lookup(GLOBAL blstrs__fp__Fp *bases, uint exponent);

DEVICE blstrs__fp__Fp blstrs__fp__Fp_mont(blstrs__fp__Fp a);
DEVICE blstrs__fp__Fp blstrs__fp__Fp_unmont(blstrs__fp__Fp a);

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool blstrs__fp__Fp_get_bit(blstrs__fp__Fp l, uint i);
// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint blstrs__fp__Fp_get_bits(blstrs__fp__Fp l, uint skip, uint window);


// Fp2 Extension Field where u^2 + 1 = 0

#define blstrs__fp2__Fp2_LIMB_BITS blstrs__fp__Fp_LIMB_BITS
#define blstrs__fp2__Fp2_ZERO ((blstrs__fp2__Fp2){blstrs__fp__Fp_ZERO, blstrs__fp__Fp_ZERO})
#define blstrs__fp2__Fp2_ONE ((blstrs__fp2__Fp2){blstrs__fp__Fp_ONE, blstrs__fp__Fp_ZERO})

typedef struct {
  blstrs__fp__Fp c0;
  blstrs__fp__Fp c1;
} blstrs__fp2__Fp2; // Represents: c0 + u * c1

DEVICE bool blstrs__fp2__Fp2_eq(blstrs__fp2__Fp2 a, blstrs__fp2__Fp2 b);
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2 a, blstrs__fp2__Fp2 b);
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_add(blstrs__fp2__Fp2 a, blstrs__fp2__Fp2 b);
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_double(blstrs__fp2__Fp2 a);
/*
 * (a_0 + u * a_1)(b_0 + u * b_1) = a_0 * b_0 - a_1 * b_1 + u * (a_0 * b_1 + a_1 * b_0)
 * Therefore:
 * c_0 = a_0 * b_0 - a_1 * b_1
 * c_1 = (a_0 * b_1 + a_1 * b_0) = (a_0 + a_1) * (b_0 + b_1) - a_0 * b_0 - a_1 * b_1
 */
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2 a, blstrs__fp2__Fp2 b);

/*
 * (a_0 + u * a_1)(a_0 + u * a_1) = a_0 ^ 2 - a_1 ^ 2 + u * 2 * a_0 * a_1
 * Therefore:
 * c_0 = (a_0 * a_0 - a_1 * a_1) = (a_0 + a_1)(a_0 - a_1)
 * c_1 = 2 * a_0 * a_1
 */
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_sqr(blstrs__fp2__Fp2 a);


/*
 * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
 */
KERNEL void blstrs__scalar__Scalar_radix_fft(GLOBAL blstrs__scalar__Scalar* x, // Source buffer
                      GLOBAL blstrs__scalar__Scalar* y, // Destination buffer
                      GLOBAL blstrs__scalar__Scalar* pq, // Precalculated twiddle factors
                      GLOBAL blstrs__scalar__Scalar* omegas, // [omega, omega^2, omega^4, ...]
                      blstrs__scalar__Scalar* u_arg, // Local buffer to store intermediary values
                      uint n, // Number of elements
                      uint lgp, // Log2 of `p` (Read more in the link above)
                      uint deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                      uint max_deg) // Maximum degree supported, according to `pq` and `omegas`
;

/// Multiplies all of the elements by `field`
KERNEL void blstrs__scalar__Scalar_mul_by_field(GLOBAL blstrs__scalar__Scalar* elements,
                        uint n,
                        blstrs__scalar__Scalar field);


// Elliptic curve operations (Short Weierstrass Jacobian form)

#define blstrs__g2__G2Affine_ZERO ((blstrs__g2__G2Affine_jacobian){blstrs__fp2__Fp2_ZERO, blstrs__fp2__Fp2_ONE, blstrs__fp2__Fp2_ZERO})

typedef struct {
  blstrs__fp2__Fp2 x;
  blstrs__fp2__Fp2 y;
} blstrs__g2__G2Affine_affine;

typedef struct {
  blstrs__fp2__Fp2 x;
  blstrs__fp2__Fp2 y;
  blstrs__fp2__Fp2 z;
} blstrs__g2__G2Affine_jacobian;

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE blstrs__g2__G2Affine_jacobian blstrs__g2__G2Affine_double(blstrs__g2__G2Affine_jacobian inp);

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE blstrs__g2__G2Affine_jacobian blstrs__g2__G2Affine_add_mixed(blstrs__g2__G2Affine_jacobian a, blstrs__g2__G2Affine_affine b);

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE blstrs__g2__G2Affine_jacobian blstrs__g2__G2Affine_add(blstrs__g2__G2Affine_jacobian a, blstrs__g2__G2Affine_jacobian b);
/*
 * Same multiexp algorithm used in Bellman, with some modifications.
 * https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L174
 * The CPU version of multiexp parallelism is done by dividing the exponent
 * values into smaller windows, and then applying a sequence of rounds to each
 * window. The GPU kernel not only assigns a thread to each window but also
 * divides the bases into several groups which highly increases the number of
 * threads running in parallel for calculating a multiexp instance.
 */

KERNEL void blstrs__g2__G2Affine_multiexp(
    GLOBAL blstrs__g2__G2Affine_affine *bases,
    GLOBAL blstrs__g2__G2Affine_jacobian *buckets,
    GLOBAL blstrs__g2__G2Affine_jacobian *results,
    GLOBAL blstrs__scalar__Scalar *exps,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size);
// Elliptic curve operations (Short Weierstrass Jacobian form)

#define blstrs__g1__G1Affine_ZERO ((blstrs__g1__G1Affine_jacobian){blstrs__fp__Fp_ZERO, blstrs__fp__Fp_ONE, blstrs__fp__Fp_ZERO})

typedef struct {
  blstrs__fp__Fp x;
  blstrs__fp__Fp y;
} blstrs__g1__G1Affine_affine;

typedef struct {
  blstrs__fp__Fp x;
  blstrs__fp__Fp y;
  blstrs__fp__Fp z;
} blstrs__g1__G1Affine_jacobian;

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE blstrs__g1__G1Affine_jacobian blstrs__g1__G1Affine_double(blstrs__g1__G1Affine_jacobian inp);

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE blstrs__g1__G1Affine_jacobian blstrs__g1__G1Affine_add_mixed(blstrs__g1__G1Affine_jacobian a, blstrs__g1__G1Affine_affine b);

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE blstrs__g1__G1Affine_jacobian blstrs__g1__G1Affine_add(blstrs__g1__G1Affine_jacobian a, blstrs__g1__G1Affine_jacobian b);
/*
 * Same multiexp algorithm used in Bellman, with some modifications.
 * https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L174
 * The CPU version of multiexp parallelism is done by dividing the exponent
 * values into smaller windows, and then applying a sequence of rounds to each
 * window. The GPU kernel not only assigns a thread to each window but also
 * divides the bases into several groups which highly increases the number of
 * threads running in parallel for calculating a multiexp instance.
 */

KERNEL void blstrs__g1__G1Affine_multiexp(
    GLOBAL blstrs__g1__G1Affine_affine *bases,
    GLOBAL blstrs__g1__G1Affine_jacobian *buckets,
    GLOBAL blstrs__g1__G1Affine_jacobian *results,
    GLOBAL blstrs__scalar__Scalar *exps,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size);

#endif
