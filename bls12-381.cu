#include "bls12-381.cuh"

CONSTANT blstrs__scalar__Scalar blstrs__scalar__Scalar_ONE = { { 4294967294, 1, 215042, 1485092858, 3971764213, 2576109551, 2898593135, 405057881 } }; // in mont
CONSTANT blstrs__scalar__Scalar blstrs__scalar__Scalar_P = { { 1, 4294967295, 4294859774, 1404937218, 161601541, 859428872, 698187080, 1944954707 } }; // not
CONSTANT blstrs__scalar__Scalar blstrs__scalar__Scalar_R2 = { { 4092763245, 3382307216, 2274516003, 728559051, 1918122383, 97719446, 2673475345, 122214873 } }; // in month
CONSTANT blstrs__scalar__Scalar blstrs__scalar__Scalar_ZERO = { { 0, 0, 0, 0, 0, 0, 0, 0 } }; // not

CONSTANT blstrs__fp__Fp blstrs__fp__Fp_ONE = { { 196605, 1980301312, 3289120770, 3958636555, 1405573306, 1598593111, 1884444485, 2010011731, 2723605613, 1543969431, 4202751123, 368467651 } };
CONSTANT blstrs__fp__Fp blstrs__fp__Fp_P = { { 4294945451, 3120496639, 2975072255, 514588670, 4138792484, 1731252896, 4085584575, 1685539716, 1129032919, 1260103606, 964683418, 436277738 } };
CONSTANT blstrs__fp__Fp blstrs__fp__Fp_R2 = { { 473175878, 4108263220, 164693233, 175564454, 1284880085, 2380613484, 2476573632, 1743489193, 3038352685, 2591637125, 2462770090, 295210981 } };
CONSTANT blstrs__fp__Fp blstrs__fp__Fp_ZERO = { { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };

// Returns a * b + c + d, puts the carry in d
DEVICE ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d) {
  #if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)
    ulong lo, hi;
    asm("mad.lo.cc.u64 %0, %2, %3, %4;\r\n"
        "madc.hi.u64 %1, %2, %3, 0;\r\n"
        "add.cc.u64 %0, %0, %5;\r\n"
        "addc.u64 %1, %1, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c), "l"(*d));
    *d = hi;
    return lo;
  #else
    ulong lo = a * b + c;
    ulong hi = mad_hi(a, b, (ulong)(lo < c));
    a = lo;
    lo += *d;
    hi += (lo < a);
    *d = hi;
    return lo;
  #endif
}

// Returns a + b, puts the carry in d
DEVICE ulong add_with_carry_64(ulong a, ulong *b) {
  #if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)
    ulong lo, hi;
    asm("add.cc.u64 %0, %2, %3;\r\n"
        "addc.u64 %1, 0, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(*b));
    *b = hi;
    return lo;
  #else
    ulong lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

// Returns a * b + c + d, puts the carry in d
DEVICE uint mac_with_carry_32(uint a, uint b, uint c, uint *d) {
  ulong res = (ulong)a * b + c + *d;
  *d = res >> 32;
  return res;
}

// Returns a + b, puts the carry in b
DEVICE uint add_with_carry_32(uint a, uint *b) {
  #if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)
    uint lo, hi;
    asm("add.cc.u32 %0, %2, %3;\r\n"
        "addc.u32 %1, 0, 0;\r\n"
        : "=r"(lo), "=r"(hi) : "r"(a), "r"(*b));
    *b = hi;
    return lo;
  #else
    uint lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

// Reverse the given bits. It's used by the FFT kernel.
DEVICE uint bitreverse(uint n, uint bits) {
  uint r = 0;
  for(int i = 0; i < bits; i++) {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  return r;
}

#ifdef BLS12_381_CUH_CUDA
// CUDA doesn't support local buffers ("dynamic shared memory" in CUDA lingo) as function
// arguments, but only a single globally defined extern value. Use `uchar` so that it is always
// allocated by the number of bytes.

DEVICE inline uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE inline uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE inline uint32_t addc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}


DEVICE inline uint32_t madlo(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madloc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhi(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline
void chain_init(chain_t *c) {
  c->_position = 0;
}

DEVICE inline
uint32_t chain_add(chain_t *ch, uint32_t a, uint32_t b) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=add_cc(a, b);
  else
    r=addc_cc(a, b);
  return r;
}

DEVICE inline
uint32_t chain_madlo(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=madlo_cc(a, b, c);
  else
    r=madloc_cc(a, b, c);
  return r;
}

DEVICE inline
uint32_t chain_madhi(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=madhi_cc(a, b, c);
  else
    r=madhic_cc(a, b, c);
  return r;
}
#endif



#if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)

DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_sub_nvidia(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) {
asm("sub.cc.u32 %0, %0, %8;\r\n"
"subc.cc.u32 %1, %1, %9;\r\n"
"subc.cc.u32 %2, %2, %10;\r\n"
"subc.cc.u32 %3, %3, %11;\r\n"
"subc.cc.u32 %4, %4, %12;\r\n"
"subc.cc.u32 %5, %5, %13;\r\n"
"subc.cc.u32 %6, %6, %14;\r\n"
"subc.u32 %7, %7, %15;\r\n"
:"+r"(a.val[0]), "+r"(a.val[1]), "+r"(a.val[2]), "+r"(a.val[3]), "+r"(a.val[4]), "+r"(a.val[5]), "+r"(a.val[6]), "+r"(a.val[7])
:"r"(b.val[0]), "r"(b.val[1]), "r"(b.val[2]), "r"(b.val[3]), "r"(b.val[4]), "r"(b.val[5]), "r"(b.val[6]), "r"(b.val[7]));
return a;
}
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_add_nvidia(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) {
asm("add.cc.u32 %0, %0, %8;\r\n"
"addc.cc.u32 %1, %1, %9;\r\n"
"addc.cc.u32 %2, %2, %10;\r\n"
"addc.cc.u32 %3, %3, %11;\r\n"
"addc.cc.u32 %4, %4, %12;\r\n"
"addc.cc.u32 %5, %5, %13;\r\n"
"addc.cc.u32 %6, %6, %14;\r\n"
"addc.u32 %7, %7, %15;\r\n"
:"+r"(a.val[0]), "+r"(a.val[1]), "+r"(a.val[2]), "+r"(a.val[3]), "+r"(a.val[4]), "+r"(a.val[5]), "+r"(a.val[6]), "+r"(a.val[7])
:"r"(b.val[0]), "r"(b.val[1]), "r"(b.val[2]), "r"(b.val[3]), "r"(b.val[4]), "r"(b.val[5]), "r"(b.val[6]), "r"(b.val[7]));
return a;
}
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

// Greater than or equal
DEVICE bool blstrs__scalar__Scalar_gte(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) {
  for(char i = blstrs__scalar__Scalar_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
DEVICE bool blstrs__scalar__Scalar_eq(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) {
  for(uchar i = 0; i < blstrs__scalar__Scalar_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)
  #define blstrs__scalar__Scalar_add_ blstrs__scalar__Scalar_add_nvidia
  #define blstrs__scalar__Scalar_sub_ blstrs__scalar__Scalar_sub_nvidia
#else
  DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_add_(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) {
    bool carry = 0;
    for(uchar i = 0; i < blstrs__scalar__Scalar_LIMBS; i++) {
      blstrs__scalar__Scalar_limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  blstrs__scalar__Scalar blstrs__scalar__Scalar_sub_(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) {
    bool borrow = 0;
    for(uchar i = 0; i < blstrs__scalar__Scalar_LIMBS; i++) {
      blstrs__scalar__Scalar_limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) {
  blstrs__scalar__Scalar res = blstrs__scalar__Scalar_sub_(a, b);
  if(!blstrs__scalar__Scalar_gte(a, b)) res = blstrs__scalar__Scalar_add_(res, blstrs__scalar__Scalar_P);
  return res;
}

// Modular addition
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_add(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) {
  blstrs__scalar__Scalar res = blstrs__scalar__Scalar_add_(a, b);
  if(blstrs__scalar__Scalar_gte(res, blstrs__scalar__Scalar_P)) res = blstrs__scalar__Scalar_sub_(res, blstrs__scalar__Scalar_P);
  return res;
}


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

DEVICE void blstrs__scalar__Scalar_reduce(uint32_t accLow[blstrs__scalar__Scalar_LIMBS], uint32_t np0, uint32_t fq[blstrs__scalar__Scalar_LIMBS]) {
  // accLow is an IN and OUT vector
  // , i must be even
  const uint32_t count = blstrs__scalar__Scalar_LIMBS;
  uint32_t accHigh[blstrs__scalar__Scalar_LIMBS];
  uint32_t bucket=0, lowCarry=0, highCarry=0, q;
  int32_t  i, j;

  #pragma unroll
  for(i=0;i<count;i++)
    accHigh[i]=0;

  // bucket is used so we don't have to push a carry all the way down the line

  #pragma unroll
  for(j=0;j<count;j++) {       // main iteration
    if(j%2==0) {
      add_cc(bucket, 0xFFFFFFFF);
      accLow[0]=addc_cc(accLow[0], accHigh[1]);
      bucket=addc(0, 0);

      q=accLow[0]*np0;

      chain_t chain1;
      chain_init(&chain1);

      #pragma unroll
      for(i=0;i<count;i+=2) {
        accLow[i]=chain_madlo(&chain1, q, fq[i], accLow[i]);
        accLow[i+1]=chain_madhi(&chain1, q, fq[i], accLow[i+1]);
      }
      lowCarry=chain_add(&chain1, 0, 0);

      chain_t chain2;
      chain_init(&chain2);
      for(i=0;i<count-2;i+=2) {
        accHigh[i]=chain_madlo(&chain2, q, fq[i+1], accHigh[i+2]);    // note the shift down
        accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], accHigh[i+3]);
      }
      accHigh[i]=chain_madlo(&chain2, q, fq[i+1], highCarry);
      accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], 0);
    }
    else {
      add_cc(bucket, 0xFFFFFFFF);
      accHigh[0]=addc_cc(accHigh[0], accLow[1]);
      bucket=addc(0, 0);

      q=accHigh[0]*np0;

      chain_t chain3;
      chain_init(&chain3);
      #pragma unroll
      for(i=0;i<count;i+=2) {
        accHigh[i]=chain_madlo(&chain3, q, fq[i], accHigh[i]);
        accHigh[i+1]=chain_madhi(&chain3, q, fq[i], accHigh[i+1]);
      }
      highCarry=chain_add(&chain3, 0, 0);

      chain_t chain4;
      chain_init(&chain4);
      for(i=0;i<count-2;i+=2) {
        accLow[i]=chain_madlo(&chain4, q, fq[i+1], accLow[i+2]);    // note the shift down
        accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], accLow[i+3]);
      }
      accLow[i]=chain_madlo(&chain4, q, fq[i+1], lowCarry);
      accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], 0);
    }
  }

  // at this point, accHigh needs to be shifted back a word and added to accLow
  // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
  // can just push it into the carry chain.

  chain_t chain5;
  chain_init(&chain5);
  chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
  #pragma unroll
  for(i=0;i<count-1;i++)
    accLow[i]=chain_add(&chain5, accLow[i], accHigh[i+1]);
  accLow[i]=chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void blstrs__scalar__Scalar_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
  const uint32_t xLimbs  = blstrs__scalar__Scalar_LIMBS;
  const uint32_t yLimbs  = blstrs__scalar__Scalar_LIMBS;
  const uint32_t xyLimbs = blstrs__scalar__Scalar_LIMBS * 2;
  uint32_t temp[blstrs__scalar__Scalar_LIMBS * 2];
  uint32_t carry = 0;

  #pragma unroll
  for (int32_t i = 0; i < xyLimbs; i++) {
    temp[i] = 0;
  }

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain1;
    chain_init(&chain1);
    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 1) {
        temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
        temp[i + j]     = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
      }
    }
    if (i % 2 == 1) {
      temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
    }
  }

  #pragma unroll
  for (int32_t i = xyLimbs - 1; i > 0; i--) {
    temp[i] = temp[i - 1];
  }
  temp[0] = 0;

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain2;
    chain_init(&chain2);

    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 0) {
        temp[i + j]     = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
        temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
      }
    }
    if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
      temp[i + yLimbs]     = chain_add(&chain2, temp[i + yLimbs], carry);
      temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
      carry = chain_add(&chain2, 0, 0);
    }
    if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
      carry = chain_add(&chain2, carry, 0);
    }
  }

  #pragma unroll
  for(int32_t i = 0; i < xyLimbs; i++) {
    xy[i] = temp[i];
  }
}

DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_mul_nvidia(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) {
  // Perform full multiply
  limb ab[2 * blstrs__scalar__Scalar_LIMBS];
  blstrs__scalar__Scalar_mult_v1(a.val, b.val, ab);

  uint32_t io[blstrs__scalar__Scalar_LIMBS];
  #pragma unroll
  for(int i=0;i<blstrs__scalar__Scalar_LIMBS;i++) {
    io[i]=ab[i];
  }
  blstrs__scalar__Scalar_reduce(io, blstrs__scalar__Scalar_INV, blstrs__scalar__Scalar_P.val);

  // Add io to the upper words of ab
  ab[blstrs__scalar__Scalar_LIMBS] = add_cc(ab[blstrs__scalar__Scalar_LIMBS], io[0]);
  int j;
  #pragma unroll
  for (j = 1; j < blstrs__scalar__Scalar_LIMBS - 1; j++) {
    ab[j + blstrs__scalar__Scalar_LIMBS] = addc_cc(ab[j + blstrs__scalar__Scalar_LIMBS], io[j]);
  }
  ab[2 * blstrs__scalar__Scalar_LIMBS - 1] = addc(ab[2 * blstrs__scalar__Scalar_LIMBS - 1], io[blstrs__scalar__Scalar_LIMBS - 1]);

  blstrs__scalar__Scalar r;
  #pragma unroll
  for (int i = 0; i < blstrs__scalar__Scalar_LIMBS; i++) {
    r.val[i] = ab[i + blstrs__scalar__Scalar_LIMBS];
  }

  if (blstrs__scalar__Scalar_gte(r, blstrs__scalar__Scalar_P)) {
    r = blstrs__scalar__Scalar_sub_(r, blstrs__scalar__Scalar_P);
  }

  return r;
}

#endif

// Modular multiplication
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_mul_default(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  blstrs__scalar__Scalar_limb t[blstrs__scalar__Scalar_LIMBS + 2] = {0};
  for(uchar i = 0; i < blstrs__scalar__Scalar_LIMBS; i++) {
    blstrs__scalar__Scalar_limb carry = 0;
    for(uchar j = 0; j < blstrs__scalar__Scalar_LIMBS; j++)
      t[j] = blstrs__scalar__Scalar_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[blstrs__scalar__Scalar_LIMBS] = blstrs__scalar__Scalar_add_with_carry(t[blstrs__scalar__Scalar_LIMBS], &carry);
    t[blstrs__scalar__Scalar_LIMBS + 1] = carry;

    carry = 0;
    blstrs__scalar__Scalar_limb m = blstrs__scalar__Scalar_INV * t[0];
    blstrs__scalar__Scalar_mac_with_carry(m, blstrs__scalar__Scalar_P.val[0], t[0], &carry);
    for(uchar j = 1; j < blstrs__scalar__Scalar_LIMBS; j++)
      t[j - 1] = blstrs__scalar__Scalar_mac_with_carry(m, blstrs__scalar__Scalar_P.val[j], t[j], &carry);

    t[blstrs__scalar__Scalar_LIMBS - 1] = blstrs__scalar__Scalar_add_with_carry(t[blstrs__scalar__Scalar_LIMBS], &carry);
    t[blstrs__scalar__Scalar_LIMBS] = t[blstrs__scalar__Scalar_LIMBS + 1] + carry;
  }

  blstrs__scalar__Scalar result;
  for(uchar i = 0; i < blstrs__scalar__Scalar_LIMBS; i++) result.val[i] = t[i];

  if(blstrs__scalar__Scalar_gte(result, blstrs__scalar__Scalar_P)) result = blstrs__scalar__Scalar_sub_(result, blstrs__scalar__Scalar_P);

  return result;
}

#ifdef BLS12_381_CUH_CUDA
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) {
  return blstrs__scalar__Scalar_mul_nvidia(a, b);
}
#else
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b) {
  return blstrs__scalar__Scalar_mul_default(a, b);
}
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_sqr(blstrs__scalar__Scalar a) {
  return blstrs__scalar__Scalar_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of blstrs__scalar__Scalar_add(a, a)
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_double(blstrs__scalar__Scalar a) {
  for(uchar i = blstrs__scalar__Scalar_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (blstrs__scalar__Scalar_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(blstrs__scalar__Scalar_gte(a, blstrs__scalar__Scalar_P)) a = blstrs__scalar__Scalar_sub_(a, blstrs__scalar__Scalar_P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_pow(blstrs__scalar__Scalar base, uint exponent) {
  blstrs__scalar__Scalar res = blstrs__scalar__Scalar_ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = blstrs__scalar__Scalar_mul(res, base);
    exponent = exponent >> 1;
    base = blstrs__scalar__Scalar_sqr(base);
  }
  return res;
}


// Store squares of the base in a lookup table for faster evaluation.
DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_pow_lookup(GLOBAL blstrs__scalar__Scalar *bases, uint exponent) {
  blstrs__scalar__Scalar res = blstrs__scalar__Scalar_ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = blstrs__scalar__Scalar_mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}

DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar a) {
  return blstrs__scalar__Scalar_mul(a, blstrs__scalar__Scalar_R2);
}

DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_unmont(blstrs__scalar__Scalar a) {
  blstrs__scalar__Scalar one = blstrs__scalar__Scalar_ZERO;
  one.val[0] = 1;
  return blstrs__scalar__Scalar_mul(a, one);
}

CONSTANT blstrs__scalar__Scalar blstrs__scalar__Scalar_P_sub_2 = { { 4294967295, 4294967294, 4294859774, 1404937218, 161601541, 859428872, 698187080, 1944954707 } };

DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_inverse(blstrs__scalar__Scalar a)
{
  blstrs__scalar__Scalar res = blstrs__scalar__Scalar_ONE;
  #pragma unroll
  for(uint i = 0; i < blstrs__scalar__Scalar_LIMBS; i++){
    #pragma unroll
    for (uint j = 0; j < blstrs__scalar__Scalar_LIMB_BITS; j++){
      if ((blstrs__scalar__Scalar_P_sub_2.val[i] >> j) & 1) res = blstrs__scalar__Scalar_mul(res, a); 
      a = blstrs__scalar__Scalar_sqr(a);         
    }
  }
  return res;
}

DEVICE blstrs__scalar__Scalar blstrs__scalar__Scalar_div(blstrs__scalar__Scalar a, blstrs__scalar__Scalar b)
{
  #pragma unroll
  for(uint i = 0; i < blstrs__scalar__Scalar_LIMBS; i++){
    #pragma unroll
    for (uint j = 0; j < blstrs__scalar__Scalar_LIMB_BITS; j++){
      if ((blstrs__scalar__Scalar_P_sub_2.val[i] >> j) & 1) a = blstrs__scalar__Scalar_mul(a, b); 
      b = blstrs__scalar__Scalar_sqr(b);         
    }
  }
  return a;
}

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool blstrs__scalar__Scalar_get_bit(blstrs__scalar__Scalar l, uint i) {
  return (l.val[blstrs__scalar__Scalar_LIMBS - 1 - i / blstrs__scalar__Scalar_LIMB_BITS] >> (blstrs__scalar__Scalar_LIMB_BITS - 1 - (i % blstrs__scalar__Scalar_LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint blstrs__scalar__Scalar_get_bits(blstrs__scalar__Scalar l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= blstrs__scalar__Scalar_get_bit(l, skip + i);
  }
  return ret;
}

#if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)

DEVICE blstrs__fp__Fp blstrs__fp__Fp_sub_nvidia(blstrs__fp__Fp a, blstrs__fp__Fp b) {
asm("sub.cc.u32 %0, %0, %12;\r\n"
"subc.cc.u32 %1, %1, %13;\r\n"
"subc.cc.u32 %2, %2, %14;\r\n"
"subc.cc.u32 %3, %3, %15;\r\n"
"subc.cc.u32 %4, %4, %16;\r\n"
"subc.cc.u32 %5, %5, %17;\r\n"
"subc.cc.u32 %6, %6, %18;\r\n"
"subc.cc.u32 %7, %7, %19;\r\n"
"subc.cc.u32 %8, %8, %20;\r\n"
"subc.cc.u32 %9, %9, %21;\r\n"
"subc.cc.u32 %10, %10, %22;\r\n"
"subc.u32 %11, %11, %23;\r\n"
:"+r"(a.val[0]), "+r"(a.val[1]), "+r"(a.val[2]), "+r"(a.val[3]), "+r"(a.val[4]), "+r"(a.val[5]), "+r"(a.val[6]), "+r"(a.val[7]), "+r"(a.val[8]), "+r"(a.val[9]), "+r"(a.val[10]), "+r"(a.val[11])
:"r"(b.val[0]), "r"(b.val[1]), "r"(b.val[2]), "r"(b.val[3]), "r"(b.val[4]), "r"(b.val[5]), "r"(b.val[6]), "r"(b.val[7]), "r"(b.val[8]), "r"(b.val[9]), "r"(b.val[10]), "r"(b.val[11]));
return a;
}
DEVICE blstrs__fp__Fp blstrs__fp__Fp_add_nvidia(blstrs__fp__Fp a, blstrs__fp__Fp b) {
asm("add.cc.u32 %0, %0, %12;\r\n"
"addc.cc.u32 %1, %1, %13;\r\n"
"addc.cc.u32 %2, %2, %14;\r\n"
"addc.cc.u32 %3, %3, %15;\r\n"
"addc.cc.u32 %4, %4, %16;\r\n"
"addc.cc.u32 %5, %5, %17;\r\n"
"addc.cc.u32 %6, %6, %18;\r\n"
"addc.cc.u32 %7, %7, %19;\r\n"
"addc.cc.u32 %8, %8, %20;\r\n"
"addc.cc.u32 %9, %9, %21;\r\n"
"addc.cc.u32 %10, %10, %22;\r\n"
"addc.u32 %11, %11, %23;\r\n"
:"+r"(a.val[0]), "+r"(a.val[1]), "+r"(a.val[2]), "+r"(a.val[3]), "+r"(a.val[4]), "+r"(a.val[5]), "+r"(a.val[6]), "+r"(a.val[7]), "+r"(a.val[8]), "+r"(a.val[9]), "+r"(a.val[10]), "+r"(a.val[11])
:"r"(b.val[0]), "r"(b.val[1]), "r"(b.val[2]), "r"(b.val[3]), "r"(b.val[4]), "r"(b.val[5]), "r"(b.val[6]), "r"(b.val[7]), "r"(b.val[8]), "r"(b.val[9]), "r"(b.val[10]), "r"(b.val[11]));
return a;
}
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

// Greater than or equal
DEVICE bool blstrs__fp__Fp_gte(blstrs__fp__Fp a, blstrs__fp__Fp b) {
  for(char i = blstrs__fp__Fp_LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
DEVICE bool blstrs__fp__Fp_eq(blstrs__fp__Fp a, blstrs__fp__Fp b) {
  for(uchar i = 0; i < blstrs__fp__Fp_LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)
  #define blstrs__fp__Fp_add_ blstrs__fp__Fp_add_nvidia
  #define blstrs__fp__Fp_sub_ blstrs__fp__Fp_sub_nvidia
#else
  DEVICE blstrs__fp__Fp blstrs__fp__Fp_add_(blstrs__fp__Fp a, blstrs__fp__Fp b) {
    bool carry = 0;
    for(uchar i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
      blstrs__fp__Fp_limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  blstrs__fp__Fp blstrs__fp__Fp_sub_(blstrs__fp__Fp a, blstrs__fp__Fp b) {
    bool borrow = 0;
    for(uchar i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
      blstrs__fp__Fp_limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
DEVICE blstrs__fp__Fp blstrs__fp__Fp_sub(blstrs__fp__Fp a, blstrs__fp__Fp b) {
  blstrs__fp__Fp res = blstrs__fp__Fp_sub_(a, b);
  if(!blstrs__fp__Fp_gte(a, b)) res = blstrs__fp__Fp_add_(res, blstrs__fp__Fp_P);
  return res;
}

// Modular addition
DEVICE blstrs__fp__Fp blstrs__fp__Fp_add(blstrs__fp__Fp a, blstrs__fp__Fp b) {
  blstrs__fp__Fp res = blstrs__fp__Fp_add_(a, b);
  if(blstrs__fp__Fp_gte(res, blstrs__fp__Fp_P)) res = blstrs__fp__Fp_sub_(res, blstrs__fp__Fp_P);
  return res;
}


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

DEVICE void blstrs__fp__Fp_reduce(uint32_t accLow[blstrs__fp__Fp_LIMBS], uint32_t np0, uint32_t fq[blstrs__fp__Fp_LIMBS]) {
  // accLow is an IN and OUT vector
  // count must be even
  const uint32_t count = blstrs__fp__Fp_LIMBS;
  uint32_t accHigh[blstrs__fp__Fp_LIMBS];
  uint32_t bucket=0, lowCarry=0, highCarry=0, q;
  int32_t  i, j;

  #pragma unroll
  for(i=0;i<count;i++)
    accHigh[i]=0;

  // bucket is used so we don't have to push a carry all the way down the line

  #pragma unroll
  for(j=0;j<count;j++) {       // main iteration
    if(j%2==0) {
      add_cc(bucket, 0xFFFFFFFF);
      accLow[0]=addc_cc(accLow[0], accHigh[1]);
      bucket=addc(0, 0);

      q=accLow[0]*np0;

      chain_t chain1;
      chain_init(&chain1);

      #pragma unroll
      for(i=0;i<count;i+=2) {
        accLow[i]=chain_madlo(&chain1, q, fq[i], accLow[i]);
        accLow[i+1]=chain_madhi(&chain1, q, fq[i], accLow[i+1]);
      }
      lowCarry=chain_add(&chain1, 0, 0);

      chain_t chain2;
      chain_init(&chain2);
      for(i=0;i<count-2;i+=2) {
        accHigh[i]=chain_madlo(&chain2, q, fq[i+1], accHigh[i+2]);    // note the shift down
        accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], accHigh[i+3]);
      }
      accHigh[i]=chain_madlo(&chain2, q, fq[i+1], highCarry);
      accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], 0);
    }
    else {
      add_cc(bucket, 0xFFFFFFFF);
      accHigh[0]=addc_cc(accHigh[0], accLow[1]);
      bucket=addc(0, 0);

      q=accHigh[0]*np0;

      chain_t chain3;
      chain_init(&chain3);
      #pragma unroll
      for(i=0;i<count;i+=2) {
        accHigh[i]=chain_madlo(&chain3, q, fq[i], accHigh[i]);
        accHigh[i+1]=chain_madhi(&chain3, q, fq[i], accHigh[i+1]);
      }
      highCarry=chain_add(&chain3, 0, 0);

      chain_t chain4;
      chain_init(&chain4);
      for(i=0;i<count-2;i+=2) {
        accLow[i]=chain_madlo(&chain4, q, fq[i+1], accLow[i+2]);    // note the shift down
        accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], accLow[i+3]);
      }
      accLow[i]=chain_madlo(&chain4, q, fq[i+1], lowCarry);
      accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], 0);
    }
  }

  // at this point, accHigh needs to be shifted back a word and added to accLow
  // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
  // can just push it into the carry chain.

  chain_t chain5;
  chain_init(&chain5);
  chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
  #pragma unroll
  for(i=0;i<count-1;i++)
    accLow[i]=chain_add(&chain5, accLow[i], accHigh[i+1]);
  accLow[i]=chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void blstrs__fp__Fp_mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
  const uint32_t xLimbs  = blstrs__fp__Fp_LIMBS;
  const uint32_t yLimbs  = blstrs__fp__Fp_LIMBS;
  const uint32_t xyLimbs = blstrs__fp__Fp_LIMBS * 2;
  uint32_t temp[blstrs__fp__Fp_LIMBS * 2];
  uint32_t carry = 0;

  #pragma unroll
  for (int32_t i = 0; i < xyLimbs; i++) {
    temp[i] = 0;
  }

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain1;
    chain_init(&chain1);
    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 1) {
        temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
        temp[i + j]     = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
      }
    }
    if (i % 2 == 1) {
      temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
    }
  }

  #pragma unroll
  for (int32_t i = xyLimbs - 1; i > 0; i--) {
    temp[i] = temp[i - 1];
  }
  temp[0] = 0;

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain2;
    chain_init(&chain2);

    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 0) {
        temp[i + j]     = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
        temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
      }
    }
    if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
      temp[i + yLimbs]     = chain_add(&chain2, temp[i + yLimbs], carry);
      temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
      carry = chain_add(&chain2, 0, 0);
    }
    if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
      carry = chain_add(&chain2, carry, 0);
    }
  }

  #pragma unroll
  for(int32_t i = 0; i < xyLimbs; i++) {
    xy[i] = temp[i];
  }
}

DEVICE blstrs__fp__Fp blstrs__fp__Fp_mul_nvidia(blstrs__fp__Fp a, blstrs__fp__Fp b) {
  // Perform full multiply
  limb ab[2 * blstrs__fp__Fp_LIMBS];
  blstrs__fp__Fp_mult_v1(a.val, b.val, ab);

  uint32_t io[blstrs__fp__Fp_LIMBS];
  #pragma unroll
  for(int i=0;i<blstrs__fp__Fp_LIMBS;i++) {
    io[i]=ab[i];
  }
  blstrs__fp__Fp_reduce(io, blstrs__fp__Fp_INV, blstrs__fp__Fp_P.val);

  // Add io to the upper words of ab
  ab[blstrs__fp__Fp_LIMBS] = add_cc(ab[blstrs__fp__Fp_LIMBS], io[0]);
  int j;
  #pragma unroll
  for (j = 1; j < blstrs__fp__Fp_LIMBS - 1; j++) {
    ab[j + blstrs__fp__Fp_LIMBS] = addc_cc(ab[j + blstrs__fp__Fp_LIMBS], io[j]);
  }
  ab[2 * blstrs__fp__Fp_LIMBS - 1] = addc(ab[2 * blstrs__fp__Fp_LIMBS - 1], io[blstrs__fp__Fp_LIMBS - 1]);

  blstrs__fp__Fp r;
  #pragma unroll
  for (int i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
    r.val[i] = ab[i + blstrs__fp__Fp_LIMBS];
  }

  if (blstrs__fp__Fp_gte(r, blstrs__fp__Fp_P)) {
    r = blstrs__fp__Fp_sub_(r, blstrs__fp__Fp_P);
  }

  return r;
}

#endif

// Modular multiplication
DEVICE blstrs__fp__Fp blstrs__fp__Fp_mul_default(blstrs__fp__Fp a, blstrs__fp__Fp b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  blstrs__fp__Fp_limb t[blstrs__fp__Fp_LIMBS + 2] = {0};
  for(uchar i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
    blstrs__fp__Fp_limb carry = 0;
    for(uchar j = 0; j < blstrs__fp__Fp_LIMBS; j++)
      t[j] = blstrs__fp__Fp_mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[blstrs__fp__Fp_LIMBS] = blstrs__fp__Fp_add_with_carry(t[blstrs__fp__Fp_LIMBS], &carry);
    t[blstrs__fp__Fp_LIMBS + 1] = carry;

    carry = 0;
    blstrs__fp__Fp_limb m = blstrs__fp__Fp_INV * t[0];
    blstrs__fp__Fp_mac_with_carry(m, blstrs__fp__Fp_P.val[0], t[0], &carry);
    for(uchar j = 1; j < blstrs__fp__Fp_LIMBS; j++)
      t[j - 1] = blstrs__fp__Fp_mac_with_carry(m, blstrs__fp__Fp_P.val[j], t[j], &carry);

    t[blstrs__fp__Fp_LIMBS - 1] = blstrs__fp__Fp_add_with_carry(t[blstrs__fp__Fp_LIMBS], &carry);
    t[blstrs__fp__Fp_LIMBS] = t[blstrs__fp__Fp_LIMBS + 1] + carry;
  }

  blstrs__fp__Fp result;
  for(uchar i = 0; i < blstrs__fp__Fp_LIMBS; i++) result.val[i] = t[i];

  if(blstrs__fp__Fp_gte(result, blstrs__fp__Fp_P)) result = blstrs__fp__Fp_sub_(result, blstrs__fp__Fp_P);

  return result;
}

#ifdef BLS12_381_CUH_CUDA
DEVICE blstrs__fp__Fp blstrs__fp__Fp_mul(blstrs__fp__Fp a, blstrs__fp__Fp b) {
  return blstrs__fp__Fp_mul_nvidia(a, b);
}
#else
DEVICE blstrs__fp__Fp blstrs__fp__Fp_mul(blstrs__fp__Fp a, blstrs__fp__Fp b) {
  return blstrs__fp__Fp_mul_default(a, b);
}
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE blstrs__fp__Fp blstrs__fp__Fp_sqr(blstrs__fp__Fp a) {
  return blstrs__fp__Fp_mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of blstrs__fp__Fp_add(a, a)
DEVICE blstrs__fp__Fp blstrs__fp__Fp_double(blstrs__fp__Fp a) {
  for(uchar i = blstrs__fp__Fp_LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (blstrs__fp__Fp_LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(blstrs__fp__Fp_gte(a, blstrs__fp__Fp_P)) a = blstrs__fp__Fp_sub_(a, blstrs__fp__Fp_P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE blstrs__fp__Fp blstrs__fp__Fp_pow(blstrs__fp__Fp base, uint exponent) {
  blstrs__fp__Fp res = blstrs__fp__Fp_ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = blstrs__fp__Fp_mul(res, base);
    exponent = exponent >> 1;
    base = blstrs__fp__Fp_sqr(base);
  }
  return res;
}


// Store squares of the base in a lookup table for faster evaluation.
DEVICE blstrs__fp__Fp blstrs__fp__Fp_pow_lookup(GLOBAL blstrs__fp__Fp *bases, uint exponent) {
  blstrs__fp__Fp res = blstrs__fp__Fp_ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = blstrs__fp__Fp_mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}

DEVICE blstrs__fp__Fp blstrs__fp__Fp_mont(blstrs__fp__Fp a) {
  return blstrs__fp__Fp_mul(a, blstrs__fp__Fp_R2);
}

DEVICE blstrs__fp__Fp blstrs__fp__Fp_unmont(blstrs__fp__Fp a) {
  blstrs__fp__Fp one = blstrs__fp__Fp_ZERO;
  one.val[0] = 1;
  return blstrs__fp__Fp_mul(a, one);
}

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool blstrs__fp__Fp_get_bit(blstrs__fp__Fp l, uint i) {
  return (l.val[blstrs__fp__Fp_LIMBS - 1 - i / blstrs__fp__Fp_LIMB_BITS] >> (blstrs__fp__Fp_LIMB_BITS - 1 - (i % blstrs__fp__Fp_LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint blstrs__fp__Fp_get_bits(blstrs__fp__Fp l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= blstrs__fp__Fp_get_bit(l, skip + i);
  }
  return ret;
}


// Fp2 Extension Field where u^2 + 1 = 0

DEVICE bool blstrs__fp2__Fp2_eq(blstrs__fp2__Fp2 a, blstrs__fp2__Fp2 b) {
  return blstrs__fp__Fp_eq(a.c0, b.c0) && blstrs__fp__Fp_eq(a.c1, b.c1);
}
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2 a, blstrs__fp2__Fp2 b) {
  a.c0 = blstrs__fp__Fp_sub(a.c0, b.c0);
  a.c1 = blstrs__fp__Fp_sub(a.c1, b.c1);
  return a;
}
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_add(blstrs__fp2__Fp2 a, blstrs__fp2__Fp2 b) {
  a.c0 = blstrs__fp__Fp_add(a.c0, b.c0);
  a.c1 = blstrs__fp__Fp_add(a.c1, b.c1);
  return a;
}
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_double(blstrs__fp2__Fp2 a) {
  a.c0 = blstrs__fp__Fp_double(a.c0);
  a.c1 = blstrs__fp__Fp_double(a.c1);
  return a;
}

/*
 * (a_0 + u * a_1)(b_0 + u * b_1) = a_0 * b_0 - a_1 * b_1 + u * (a_0 * b_1 + a_1 * b_0)
 * Therefore:
 * c_0 = a_0 * b_0 - a_1 * b_1
 * c_1 = (a_0 * b_1 + a_1 * b_0) = (a_0 + a_1) * (b_0 + b_1) - a_0 * b_0 - a_1 * b_1
 */
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2 a, blstrs__fp2__Fp2 b) {
  const blstrs__fp__Fp aa = blstrs__fp__Fp_mul(a.c0, b.c0);
  const blstrs__fp__Fp bb = blstrs__fp__Fp_mul(a.c1, b.c1);
  const blstrs__fp__Fp o = blstrs__fp__Fp_add(b.c0, b.c1);
  a.c1 = blstrs__fp__Fp_add(a.c1, a.c0);
  a.c1 = blstrs__fp__Fp_mul(a.c1, o);
  a.c1 = blstrs__fp__Fp_sub(a.c1, aa);
  a.c1 = blstrs__fp__Fp_sub(a.c1, bb);
  a.c0 = blstrs__fp__Fp_sub(aa, bb);
  return a;
}

/*
 * (a_0 + u * a_1)(a_0 + u * a_1) = a_0 ^ 2 - a_1 ^ 2 + u * 2 * a_0 * a_1
 * Therefore:
 * c_0 = (a_0 * a_0 - a_1 * a_1) = (a_0 + a_1)(a_0 - a_1)
 * c_1 = 2 * a_0 * a_1
 */
DEVICE blstrs__fp2__Fp2 blstrs__fp2__Fp2_sqr(blstrs__fp2__Fp2 a) {
  const blstrs__fp__Fp ab = blstrs__fp__Fp_mul(a.c0, a.c1);
  const blstrs__fp__Fp c0c1 = blstrs__fp__Fp_add(a.c0, a.c1);
  a.c0 = blstrs__fp__Fp_mul(blstrs__fp__Fp_sub(a.c0, a.c1), c0c1);
  a.c1 = blstrs__fp__Fp_double(ab);
  return a;
}


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
{
// CUDA doesn't support local buffers ("shared memory" in CUDA lingo) as function arguments,
// ignore that argument and use the globally defined extern memory instead.
#ifdef BLS12_381_CUH_CUDA
  // There can only be a single dynamic shared memory item, hence cast it to the type we need.
  blstrs__scalar__Scalar* u = (blstrs__scalar__Scalar*)cuda_shared;
#else
  LOCAL blstrs__scalar__Scalar* u = u_arg;
#endif

  uint lid = GET_LOCAL_ID();
  uint lsize = GET_LOCAL_SIZE();
  uint index = GET_GROUP_ID();
  uint t = n >> deg;
  uint p = 1 << lgp;
  uint k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint count = 1 << deg; // 2^deg
  uint counth = count >> 1; // Half of count

  uint counts = count / lsize * lid;
  uint counte = counts + count / lsize;

  // Compute powers of twiddle
  const blstrs__scalar__Scalar twiddle = blstrs__scalar__Scalar_pow_lookup(omegas, (n >> lgp >> deg) * k);
  blstrs__scalar__Scalar tmp = blstrs__scalar__Scalar_pow(twiddle, counts);
  for(uint i = counts; i < counte; i++) {
    u[i] = blstrs__scalar__Scalar_mul(tmp, x[i*t]);
    tmp = blstrs__scalar__Scalar_mul(tmp, twiddle);
  }
  BARRIER_LOCAL();

  const uint pqshift = max_deg - deg;
  for(uint rnd = 0; rnd < deg; rnd++) {
    const uint bit = counth >> rnd;
    for(uint i = counts >> 1; i < counte >> 1; i++) {
      const uint di = i & (bit - 1);
      const uint i0 = (i << 1) - di;
      const uint i1 = i0 + bit;
      tmp = u[i0];
      u[i0] = blstrs__scalar__Scalar_add(u[i0], u[i1]);
      u[i1] = blstrs__scalar__Scalar_sub(tmp, u[i1]);
      if(di != 0) u[i1] = blstrs__scalar__Scalar_mul(pq[di << rnd << pqshift], u[i1]);
    }

    BARRIER_LOCAL();
  }

  for(uint i = counts >> 1; i < counte >> 1; i++) {
    y[i*p] = u[bitreverse(i, deg)];
    y[(i+counth)*p] = u[bitreverse(i + counth, deg)];
  }
}

/// Multiplies all of the elements by `field`
KERNEL void blstrs__scalar__Scalar_mul_by_field(GLOBAL blstrs__scalar__Scalar* elements,
                        uint n,
                        blstrs__scalar__Scalar field) {
  const uint gid = GET_GLOBAL_ID();
  elements[gid] = blstrs__scalar__Scalar_mul(elements[gid], field);
}


// Elliptic curve operations (Short Weierstrass Jacobian form)


// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE blstrs__g2__G2Affine_jacobian blstrs__g2__G2Affine_double(blstrs__g2__G2Affine_jacobian inp) {
  const blstrs__fp2__Fp2 local_zero = blstrs__fp2__Fp2_ZERO;
  if(blstrs__fp2__Fp2_eq(inp.z, local_zero)) {
      return inp;
  }

  const blstrs__fp2__Fp2 a = blstrs__fp2__Fp2_sqr(inp.x); // A = X1^2
  const blstrs__fp2__Fp2 b = blstrs__fp2__Fp2_sqr(inp.y); // B = Y1^2
  blstrs__fp2__Fp2 c = blstrs__fp2__Fp2_sqr(b); // C = B^2

  // D = 2*((X1+B)2-A-C)
  blstrs__fp2__Fp2 d = blstrs__fp2__Fp2_add(inp.x, b);
  d = blstrs__fp2__Fp2_sqr(d); d = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(d, a), c); d = blstrs__fp2__Fp2_double(d);

  const blstrs__fp2__Fp2 e = blstrs__fp2__Fp2_add(blstrs__fp2__Fp2_double(a), a); // E = 3*A
  const blstrs__fp2__Fp2 f = blstrs__fp2__Fp2_sqr(e);

  inp.z = blstrs__fp2__Fp2_mul(inp.y, inp.z); inp.z = blstrs__fp2__Fp2_double(inp.z); // Z3 = 2*Y1*Z1
  inp.x = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(f, d), d); // X3 = F-2*D

  // Y3 = E*(D-X3)-8*C
  c = blstrs__fp2__Fp2_double(c); c = blstrs__fp2__Fp2_double(c); c = blstrs__fp2__Fp2_double(c);
  inp.y = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2_sub(d, inp.x), e), c);

  return inp;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE blstrs__g2__G2Affine_jacobian blstrs__g2__G2Affine_add_mixed(blstrs__g2__G2Affine_jacobian a, blstrs__g2__G2Affine_affine b) {
  const blstrs__fp2__Fp2 local_zero = blstrs__fp2__Fp2_ZERO;
  if(blstrs__fp2__Fp2_eq(a.z, local_zero)) {
    const blstrs__fp2__Fp2 local_one = blstrs__fp2__Fp2_ONE;
    a.x = b.x;
    a.y = b.y;
    a.z = local_one;
    return a;
  }

  const blstrs__fp2__Fp2 z1z1 = blstrs__fp2__Fp2_sqr(a.z);
  const blstrs__fp2__Fp2 u2 = blstrs__fp2__Fp2_mul(b.x, z1z1);
  const blstrs__fp2__Fp2 s2 = blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2_mul(b.y, a.z), z1z1);

  if(blstrs__fp2__Fp2_eq(a.x, u2) && blstrs__fp2__Fp2_eq(a.y, s2)) {
      return blstrs__g2__G2Affine_double(a);
  }

  const blstrs__fp2__Fp2 h = blstrs__fp2__Fp2_sub(u2, a.x); // H = U2-X1
  const blstrs__fp2__Fp2 hh = blstrs__fp2__Fp2_sqr(h); // HH = H^2
  blstrs__fp2__Fp2 i = blstrs__fp2__Fp2_double(hh); i = blstrs__fp2__Fp2_double(i); // I = 4*HH
  blstrs__fp2__Fp2 j = blstrs__fp2__Fp2_mul(h, i); // J = H*I
  blstrs__fp2__Fp2 r = blstrs__fp2__Fp2_sub(s2, a.y); r = blstrs__fp2__Fp2_double(r); // r = 2*(S2-Y1)
  const blstrs__fp2__Fp2 v = blstrs__fp2__Fp2_mul(a.x, i);

  blstrs__g2__G2Affine_jacobian ret;

  // X3 = r^2 - J - 2*V
  ret.x = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sqr(r), j), blstrs__fp2__Fp2_double(v));

  // Y3 = r*(V-X3)-2*Y1*J
  j = blstrs__fp2__Fp2_mul(a.y, j); j = blstrs__fp2__Fp2_double(j);
  ret.y = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2_sub(v, ret.x), r), j);

  // Z3 = (Z1+H)^2-Z1Z1-HH
  ret.z = blstrs__fp2__Fp2_add(a.z, h); ret.z = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sqr(ret.z), z1z1), hh);
  return ret;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE blstrs__g2__G2Affine_jacobian blstrs__g2__G2Affine_add(blstrs__g2__G2Affine_jacobian a, blstrs__g2__G2Affine_jacobian b) {

  const blstrs__fp2__Fp2 local_zero = blstrs__fp2__Fp2_ZERO;
  if(blstrs__fp2__Fp2_eq(a.z, local_zero)) return b;
  if(blstrs__fp2__Fp2_eq(b.z, local_zero)) return a;

  const blstrs__fp2__Fp2 z1z1 = blstrs__fp2__Fp2_sqr(a.z); // Z1Z1 = Z1^2
  const blstrs__fp2__Fp2 z2z2 = blstrs__fp2__Fp2_sqr(b.z); // Z2Z2 = Z2^2
  const blstrs__fp2__Fp2 u1 = blstrs__fp2__Fp2_mul(a.x, z2z2); // U1 = X1*Z2Z2
  const blstrs__fp2__Fp2 u2 = blstrs__fp2__Fp2_mul(b.x, z1z1); // U2 = X2*Z1Z1
  blstrs__fp2__Fp2 s1 = blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2_mul(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
  const blstrs__fp2__Fp2 s2 = blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2_mul(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

  if(blstrs__fp2__Fp2_eq(u1, u2) && blstrs__fp2__Fp2_eq(s1, s2))
    return blstrs__g2__G2Affine_double(a);
  else {
    const blstrs__fp2__Fp2 h = blstrs__fp2__Fp2_sub(u2, u1); // H = U2-U1
    blstrs__fp2__Fp2 i = blstrs__fp2__Fp2_double(h); i = blstrs__fp2__Fp2_sqr(i); // I = (2*H)^2
    const blstrs__fp2__Fp2 j = blstrs__fp2__Fp2_mul(h, i); // J = H*I
    blstrs__fp2__Fp2 r = blstrs__fp2__Fp2_sub(s2, s1); r = blstrs__fp2__Fp2_double(r); // r = 2*(S2-S1)
    const blstrs__fp2__Fp2 v = blstrs__fp2__Fp2_mul(u1, i); // V = U1*I
    a.x = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sqr(r), j), v), v); // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*S1*J
    a.y = blstrs__fp2__Fp2_mul(blstrs__fp2__Fp2_sub(v, a.x), r);
    s1 = blstrs__fp2__Fp2_mul(s1, j); s1 = blstrs__fp2__Fp2_double(s1); // S1 = S1 * J * 2
    a.y = blstrs__fp2__Fp2_sub(a.y, s1);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    a.z = blstrs__fp2__Fp2_add(a.z, b.z); a.z = blstrs__fp2__Fp2_sqr(a.z);
    a.z = blstrs__fp2__Fp2_sub(blstrs__fp2__Fp2_sub(a.z, z1z1), z2z2);
    a.z = blstrs__fp2__Fp2_mul(a.z, h);

    return a;
  }
}
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
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  // We have (2^window_size - 1) buckets.
  const uint bucket_len = ((1 << window_size) - 1);

  // Each thread has its own set of buckets in global memory.
  buckets += bucket_len * gid;

  const blstrs__g2__G2Affine_jacobian local_zero = blstrs__g2__G2Affine_ZERO;
  for(uint i = 0; i < bucket_len; i++) buckets[i] = local_zero;

  // Num of elements in each group. Round the number up (ceil).
  const uint len = (n + num_groups - 1) / num_groups;

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  const uint nstart = len * (gid / num_windows);
  const uint nend = min(nstart + len, n);
  const uint bits = (gid % num_windows) * window_size;
  const ushort w = min((ushort)window_size, (ushort)(blstrs__scalar__Scalar_BITS - bits));

  blstrs__g2__G2Affine_jacobian res = blstrs__g2__G2Affine_ZERO;
  for(uint i = nstart; i < nend; i++) {
    uint ind = blstrs__scalar__Scalar_get_bits(exps[i], bits, w);

    #if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)
      // O_o, weird optimization, having a single special case makes it
      // tremendously faster!
      // 511 is chosen because it's half of the maximum bucket len, but
      // any other number works... Bigger indices seems to be better...
      if(ind == 511) buckets[510] = blstrs__g2__G2Affine_add_mixed(buckets[510], bases[i]);
      else if(ind--) buckets[ind] = blstrs__g2__G2Affine_add_mixed(buckets[ind], bases[i]);
    #else
      if(ind--) buckets[ind] = blstrs__g2__G2Affine_add_mixed(buckets[ind], bases[i]);
    #endif
  }

  // Summation by parts
  // e.g. 3a + 2b + 1c = a +
  //                    (a) + b +
  //                    ((a) + b) + c
  blstrs__g2__G2Affine_jacobian acc = blstrs__g2__G2Affine_ZERO;
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = blstrs__g2__G2Affine_add(acc, buckets[j]);
    res = blstrs__g2__G2Affine_add(res, acc);
  }

  results[gid] = res;
}
// Elliptic curve operations (Short Weierstrass Jacobian form)

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE blstrs__g1__G1Affine_jacobian blstrs__g1__G1Affine_double(blstrs__g1__G1Affine_jacobian inp) {
  const blstrs__fp__Fp local_zero = blstrs__fp__Fp_ZERO;
  if(blstrs__fp__Fp_eq(inp.z, local_zero)) {
      return inp;
  }

  const blstrs__fp__Fp a = blstrs__fp__Fp_sqr(inp.x); // A = X1^2
  const blstrs__fp__Fp b = blstrs__fp__Fp_sqr(inp.y); // B = Y1^2
  blstrs__fp__Fp c = blstrs__fp__Fp_sqr(b); // C = B^2

  // D = 2*((X1+B)2-A-C)
  blstrs__fp__Fp d = blstrs__fp__Fp_add(inp.x, b);
  d = blstrs__fp__Fp_sqr(d); d = blstrs__fp__Fp_sub(blstrs__fp__Fp_sub(d, a), c); d = blstrs__fp__Fp_double(d);

  const blstrs__fp__Fp e = blstrs__fp__Fp_add(blstrs__fp__Fp_double(a), a); // E = 3*A
  const blstrs__fp__Fp f = blstrs__fp__Fp_sqr(e);

  inp.z = blstrs__fp__Fp_mul(inp.y, inp.z); inp.z = blstrs__fp__Fp_double(inp.z); // Z3 = 2*Y1*Z1
  inp.x = blstrs__fp__Fp_sub(blstrs__fp__Fp_sub(f, d), d); // X3 = F-2*D

  // Y3 = E*(D-X3)-8*C
  c = blstrs__fp__Fp_double(c); c = blstrs__fp__Fp_double(c); c = blstrs__fp__Fp_double(c);
  inp.y = blstrs__fp__Fp_sub(blstrs__fp__Fp_mul(blstrs__fp__Fp_sub(d, inp.x), e), c);

  return inp;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE blstrs__g1__G1Affine_jacobian blstrs__g1__G1Affine_add_mixed(blstrs__g1__G1Affine_jacobian a, blstrs__g1__G1Affine_affine b) {
  const blstrs__fp__Fp local_zero = blstrs__fp__Fp_ZERO;
  if(blstrs__fp__Fp_eq(a.z, local_zero)) {
    const blstrs__fp__Fp local_one = blstrs__fp__Fp_ONE;
    a.x = b.x;
    a.y = b.y;
    a.z = local_one;
    return a;
  }

  const blstrs__fp__Fp z1z1 = blstrs__fp__Fp_sqr(a.z);
  const blstrs__fp__Fp u2 = blstrs__fp__Fp_mul(b.x, z1z1);
  const blstrs__fp__Fp s2 = blstrs__fp__Fp_mul(blstrs__fp__Fp_mul(b.y, a.z), z1z1);

  if(blstrs__fp__Fp_eq(a.x, u2) && blstrs__fp__Fp_eq(a.y, s2)) {
      return blstrs__g1__G1Affine_double(a);
  }

  const blstrs__fp__Fp h = blstrs__fp__Fp_sub(u2, a.x); // H = U2-X1
  const blstrs__fp__Fp hh = blstrs__fp__Fp_sqr(h); // HH = H^2
  blstrs__fp__Fp i = blstrs__fp__Fp_double(hh); i = blstrs__fp__Fp_double(i); // I = 4*HH
  blstrs__fp__Fp j = blstrs__fp__Fp_mul(h, i); // J = H*I
  blstrs__fp__Fp r = blstrs__fp__Fp_sub(s2, a.y); r = blstrs__fp__Fp_double(r); // r = 2*(S2-Y1)
  const blstrs__fp__Fp v = blstrs__fp__Fp_mul(a.x, i);

  blstrs__g1__G1Affine_jacobian ret;

  // X3 = r^2 - J - 2*V
  ret.x = blstrs__fp__Fp_sub(blstrs__fp__Fp_sub(blstrs__fp__Fp_sqr(r), j), blstrs__fp__Fp_double(v));

  // Y3 = r*(V-X3)-2*Y1*J
  j = blstrs__fp__Fp_mul(a.y, j); j = blstrs__fp__Fp_double(j);
  ret.y = blstrs__fp__Fp_sub(blstrs__fp__Fp_mul(blstrs__fp__Fp_sub(v, ret.x), r), j);

  // Z3 = (Z1+H)^2-Z1Z1-HH
  ret.z = blstrs__fp__Fp_add(a.z, h); ret.z = blstrs__fp__Fp_sub(blstrs__fp__Fp_sub(blstrs__fp__Fp_sqr(ret.z), z1z1), hh);
  return ret;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE blstrs__g1__G1Affine_jacobian blstrs__g1__G1Affine_add(blstrs__g1__G1Affine_jacobian a, blstrs__g1__G1Affine_jacobian b) {

  const blstrs__fp__Fp local_zero = blstrs__fp__Fp_ZERO;
  if(blstrs__fp__Fp_eq(a.z, local_zero)) return b;
  if(blstrs__fp__Fp_eq(b.z, local_zero)) return a;

  const blstrs__fp__Fp z1z1 = blstrs__fp__Fp_sqr(a.z); // Z1Z1 = Z1^2
  const blstrs__fp__Fp z2z2 = blstrs__fp__Fp_sqr(b.z); // Z2Z2 = Z2^2
  const blstrs__fp__Fp u1 = blstrs__fp__Fp_mul(a.x, z2z2); // U1 = X1*Z2Z2
  const blstrs__fp__Fp u2 = blstrs__fp__Fp_mul(b.x, z1z1); // U2 = X2*Z1Z1
  blstrs__fp__Fp s1 = blstrs__fp__Fp_mul(blstrs__fp__Fp_mul(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
  const blstrs__fp__Fp s2 = blstrs__fp__Fp_mul(blstrs__fp__Fp_mul(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

  if(blstrs__fp__Fp_eq(u1, u2) && blstrs__fp__Fp_eq(s1, s2))
    return blstrs__g1__G1Affine_double(a);
  else {
    const blstrs__fp__Fp h = blstrs__fp__Fp_sub(u2, u1); // H = U2-U1
    blstrs__fp__Fp i = blstrs__fp__Fp_double(h); i = blstrs__fp__Fp_sqr(i); // I = (2*H)^2
    const blstrs__fp__Fp j = blstrs__fp__Fp_mul(h, i); // J = H*I
    blstrs__fp__Fp r = blstrs__fp__Fp_sub(s2, s1); r = blstrs__fp__Fp_double(r); // r = 2*(S2-S1)
    const blstrs__fp__Fp v = blstrs__fp__Fp_mul(u1, i); // V = U1*I
    a.x = blstrs__fp__Fp_sub(blstrs__fp__Fp_sub(blstrs__fp__Fp_sub(blstrs__fp__Fp_sqr(r), j), v), v); // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*S1*J
    a.y = blstrs__fp__Fp_mul(blstrs__fp__Fp_sub(v, a.x), r);
    s1 = blstrs__fp__Fp_mul(s1, j); s1 = blstrs__fp__Fp_double(s1); // S1 = S1 * J * 2
    a.y = blstrs__fp__Fp_sub(a.y, s1);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    a.z = blstrs__fp__Fp_add(a.z, b.z); a.z = blstrs__fp__Fp_sqr(a.z);
    a.z = blstrs__fp__Fp_sub(blstrs__fp__Fp_sub(a.z, z1z1), z2z2);
    a.z = blstrs__fp__Fp_mul(a.z, h);

    return a;
  }
}
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
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  // We have (2^window_size - 1) buckets.
  const uint bucket_len = ((1 << window_size) - 1);

  // Each thread has its own set of buckets in global memory.
  buckets += bucket_len * gid;

  const blstrs__g1__G1Affine_jacobian local_zero = blstrs__g1__G1Affine_ZERO;
  for(uint i = 0; i < bucket_len; i++) buckets[i] = local_zero;

  // Num of elements in each group. Round the number up (ceil).
  const uint len = (n + num_groups - 1) / num_groups;

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  const uint nstart = len * (gid / num_windows);
  const uint nend = min(nstart + len, n);
  const uint bits = (gid % num_windows) * window_size;
  const ushort w = min((ushort)window_size, (ushort)(blstrs__scalar__Scalar_BITS - bits));

  blstrs__g1__G1Affine_jacobian res = blstrs__g1__G1Affine_ZERO;
  for(uint i = nstart; i < nend; i++) {
    uint ind = blstrs__scalar__Scalar_get_bits(exps[i], bits, w);

    #if defined(BLS12_381_CUH_OPENCL_NVIDIA) || defined(BLS12_381_CUH_CUDA)
      // O_o, weird optimization, having a single special case makes it
      // tremendously faster!
      // 511 is chosen because it's half of the maximum bucket len, but
      // any other number works... Bigger indices seems to be better...
      if(ind == 511) buckets[510] = blstrs__g1__G1Affine_add_mixed(buckets[510], bases[i]);
      else if(ind--) buckets[ind] = blstrs__g1__G1Affine_add_mixed(buckets[ind], bases[i]);
    #else
      if(ind--) buckets[ind] = blstrs__g1__G1Affine_add_mixed(buckets[ind], bases[i]);
    #endif
  }

  // Summation by parts
  // e.g. 3a + 2b + 1c = a +
  //                    (a) + b +
  //                    ((a) + b) + c
  blstrs__g1__G1Affine_jacobian acc = blstrs__g1__G1Affine_ZERO;
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = blstrs__g1__G1Affine_add(acc, buckets[j]);
    res = blstrs__g1__G1Affine_add(res, acc);
  }

  results[gid] = res;
}


