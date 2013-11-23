
#if defined(SUPERCOP)
  #include "crypto_stream.h"
#endif

#include <stdint.h>
#include <string.h>

#include <immintrin.h>

/* Default to 20 rounds */
#if !defined(CHACHA_ROUNDS)
  #define CHACHA_ROUNDS 20
#endif

#if 0 != (CHACHA_ROUNDS % 2)
  #error "I only accept an even number of rounds!"
#endif

/* Some aliases to keep code below somewhat sane */
#define LOADU(m)     _mm256_loadu_si256((const __m256i *)(m))
#define STOREU(m, v) _mm256_storeu_si256((__m256i*)(m), (v))
#define TO128(x)     _mm256_castsi256_si128(x)

#define ADD(A, B) _mm256_add_epi32(A, B)
#define SUB(A, B) _mm256_sub_epi32(A, B)
#define XOR(A, B) _mm256_xor_si256(A, B)
#define ROT(X, C)                                                                                    \
(                                                                                                    \
        (C) ==  8 ? _mm256_shuffle_epi8((X), _mm256_set_epi8(14,13,12,15,10,9,8,11,6,5,4,7,2,1,0,3,  \
                                                             14,13,12,15,10,9,8,11,6,5,4,7,2,1,0,3)) \
    :   (C) == 16 ? _mm256_shuffle_epi8((X), _mm256_set_epi8(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2,  \
                                                             13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2)) \
    :   /* else */  _mm256_or_si256(_mm256_slli_epi32((X), (C)), _mm256_srli_epi32((X), 32 - (C)))   \
)

#if defined(MANUAL_SCHEDULING) /* Behaves better on clang 3.3, worse on gcc, icc */
#define DOUBLE_ROUND(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15)           \
do                                                                                    \
{                                                                                     \
  /* Column round */                                                                  \
   x0 = ADD( x0,  x4);  x1 = ADD( x1,  x5);  x2 = ADD( x2,  x6);  x3 = ADD( x3,  x7); \
  x12 = XOR(x12,  x0); x13 = XOR(x13,  x1); x14 = XOR(x14,  x2); x15 = XOR(x15,  x3); \
  x12 = ROT(x12,  16); x13 = ROT(x13,  16); x14 = ROT(x14,  16); x15 = ROT(x15,  16); \
   x8 = ADD( x8, x12);  x9 = ADD( x9, x13); x10 = ADD(x10, x14); x11 = ADD(x11, x15); \
   x4 = XOR( x4,  x8);  x5 = XOR( x5,  x9);  x6 = XOR( x6, x10);  x7 = XOR( x7, x11); \
   x4 = ROT( x4,  12);  x5 = ROT( x5,  12);  x6 = ROT( x6,  12);  x7 = ROT( x7,  12); \
   x0 = ADD( x0,  x4);  x1 = ADD( x1,  x5);  x2 = ADD( x2,  x6);  x3 = ADD( x3,  x7); \
  x12 = XOR(x12,  x0); x13 = XOR(x13,  x1); x14 = XOR(x14,  x2); x15 = XOR(x15,  x3); \
  x12 = ROT(x12,   8); x13 = ROT(x13,   8); x14 = ROT(x14,   8); x15 = ROT(x15,   8); \
   x8 = ADD( x8, x12);  x9 = ADD( x9, x13); x10 = ADD(x10, x14); x11 = ADD(x11, x15); \
   x4 = XOR( x4,  x8);  x5 = XOR( x5,  x9);  x6 = XOR( x6, x10);  x7 = XOR( x7, x11); \
   x4 = ROT( x4,   7);  x5 = ROT( x5,   7);  x6 = ROT( x6,   7);  x7 = ROT( x7,   7); \
   /* Diagonal round */                                                               \
   x0 = ADD( x0,  x5);  x1 = ADD( x1,  x6);  x2 = ADD( x2,  x7);  x3 = ADD( x3,  x4); \
  x15 = XOR(x15,  x0); x12 = XOR(x12,  x1); x13 = XOR(x13,  x2); x14 = XOR(x14,  x3); \
  x15 = ROT(x15,  16); x12 = ROT(x12,  16); x13 = ROT(x13,  16); x14 = ROT(x14,  16); \
  x10 = ADD(x10, x15); x11 = ADD(x11, x12);  x8 = ADD( x8, x13);  x9 = ADD( x9, x14); \
   x5 = XOR( x5, x10);  x6 = XOR( x6, x11);  x7 = XOR( x7,  x8);  x4 = XOR( x4,  x9); \
   x5 = ROT( x5,  12);  x6 = ROT( x6,  12);  x7 = ROT( x7,  12);  x4 = ROT( x4,  12); \
   x0 = ADD( x0,  x5);  x1 = ADD( x1,  x6);  x2 = ADD( x2,  x7);  x3 = ADD( x3,  x4); \
  x15 = XOR(x15,  x0); x12 = XOR(x12,  x1); x13 = XOR(x13,  x2); x14 = XOR(x14,  x3); \
  x15 = ROT(x15,   8); x12 = ROT(x12,   8); x13 = ROT(x13,   8); x14 = ROT(x14,   8); \
  x10 = ADD(x10, x15); x11 = ADD(x11, x12);  x8 = ADD( x8, x13);  x9 = ADD( x9, x14); \
   x5 = XOR( x5, x10);  x6 = XOR( x6, x11);  x7 = XOR( x7,  x8);  x4 = XOR( x4,  x9); \
   x5 = ROT( x5,   7);  x6 = ROT( x6,   7);  x7 = ROT( x7,   7);  x4 = ROT( x4,   7); \
} while(0)
#else
#define QUARTER_ROUND(A, B, C, D)                     \
do                                                    \
{                                                     \
  A  = ADD(A, B); D  = XOR(D, A); D  = ROT(D, 16);    \
  C  = ADD(C, D); B  = XOR(B, C); B  = ROT(B, 12);    \
  A  = ADD(A, B); D  = XOR(D, A); D  = ROT(D,  8);    \
  C  = ADD(C, D); B  = XOR(B, C); B  = ROT(B,  7);    \
} while(0)

#define DOUBLE_ROUND(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15) \
do                                                                          \
{                                                                           \
  /* Column round */                                                        \
  QUARTER_ROUND(x0, x4,  x8, x12);                                          \
  QUARTER_ROUND(x1, x5,  x9, x13);                                          \
  QUARTER_ROUND(x2, x6, x10, x14);                                          \
  QUARTER_ROUND(x3, x7, x11, x15);                                          \
  /* Diagonal round */                                                      \
  QUARTER_ROUND(x0, x5, x10, x15);                                          \
  QUARTER_ROUND(x1, x6, x11, x12);                                          \
  QUARTER_ROUND(x2, x7,  x8, x13);                                          \
  QUARTER_ROUND(x3, x4,  x9, x14);                                          \
} while(0)
#endif

#define CHACHA_CORE(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15, \
                    x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15) \
do                                                                         \
{                                                                          \
  int i;                                                                   \
   z0 =  x0, z1 =  x1, z2 =  x2, z3 =  x3,                                 \
   z4 =  x4, z5 =  x5, z6 =  x6, z7 =  x7,                                 \
   z8 =  x8, z9 =  x9,z10 = x10,z11 = x11,                                 \
  z12 = x12,z13 = x13,z14 = x14,z15 = x15;                                 \
  for(i = 0; i < CHACHA_ROUNDS; i += 2)                                    \
    DOUBLE_ROUND(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15);   \
                                                                           \
  z0  = ADD( x0,  z0);                                                     \
  z1  = ADD( x1,  z1);                                                     \
  z2  = ADD( x2,  z2);                                                     \
  z3  = ADD( x3,  z3);                                                     \
  z4  = ADD( x4,  z4);                                                     \
  z5  = ADD( x5,  z5);                                                     \
  z6  = ADD( x6,  z6);                                                     \
  z7  = ADD( x7,  z7);                                                     \
  z8  = ADD( x8,  z8);                                                     \
  z9  = ADD( x9,  z9);                                                     \
  z10 = ADD(x10, z10);                                                     \
  z11 = ADD(x11, z11);                                                     \
  z12 = ADD(x12, z12);                                                     \
  z13 = ADD(x13, z13);                                                     \
  z14 = ADD(x14, z14);                                                     \
  z15 = ADD(x15, z15);                                                     \
} while(0)


/* Probably not the best it can be... */
#define TRANSPOSE(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15)  \
do                                                                        \
{                                                                         \
  __m256i t0,t1,t2,t3,t4,t5,t6,t7;                                        \
  __m256i t8,t9,t10,t11,t12,t13,t14,t15;                                  \
  __m256i s0,s1,s2,s3,s4,s5,s6,s7;                                        \
  __m256i s8,s9,s10,s11,s12,s13,s14,s15;                                  \
                                                                          \
  t0  = _mm256_unpacklo_epi32( x0,  x1);                                  \
  t1  = _mm256_unpackhi_epi32( x0,  x1);                                  \
  t2  = _mm256_unpacklo_epi32( x2,  x3);                                  \
  t3  = _mm256_unpackhi_epi32( x2,  x3);                                  \
  t4  = _mm256_unpacklo_epi32( x4,  x5);                                  \
  t5  = _mm256_unpackhi_epi32( x4,  x5);                                  \
  t6  = _mm256_unpacklo_epi32( x6,  x7);                                  \
  t7  = _mm256_unpackhi_epi32( x6,  x7);                                  \
  t8  = _mm256_unpacklo_epi32( x8,  x9);                                  \
  t9  = _mm256_unpackhi_epi32( x8,  x9);                                  \
  t10 = _mm256_unpacklo_epi32(x10, x11);                                  \
  t11 = _mm256_unpackhi_epi32(x10, x11);                                  \
  t12 = _mm256_unpacklo_epi32(x12, x13);                                  \
  t13 = _mm256_unpackhi_epi32(x12, x13);                                  \
  t14 = _mm256_unpacklo_epi32(x14, x15);                                  \
  t15 = _mm256_unpackhi_epi32(x14, x15);                                  \
                                                                          \
  s0  = _mm256_unpacklo_epi64( t0,  t2);                                  \
  s1  = _mm256_unpackhi_epi64( t0,  t2);                                  \
  s2  = _mm256_unpacklo_epi64( t1,  t3);                                  \
  s3  = _mm256_unpackhi_epi64( t1,  t3);                                  \
  s4  = _mm256_unpacklo_epi64( t4,  t6);                                  \
  s5  = _mm256_unpackhi_epi64( t4,  t6);                                  \
  s6  = _mm256_unpacklo_epi64( t5,  t7);                                  \
  s7  = _mm256_unpackhi_epi64( t5,  t7);                                  \
  s8  = _mm256_unpacklo_epi64( t8, t10);                                  \
  s9  = _mm256_unpackhi_epi64( t8, t10);                                  \
  s10 = _mm256_unpacklo_epi64( t9, t11);                                  \
  s11 = _mm256_unpackhi_epi64( t9, t11);                                  \
  s12 = _mm256_unpacklo_epi64(t12, t14);                                  \
  s13 = _mm256_unpackhi_epi64(t12, t14);                                  \
  s14 = _mm256_unpacklo_epi64(t13, t15);                                  \
  s15 = _mm256_unpackhi_epi64(t13, t15);                                  \
                                                                          \
  x0  = _mm256_permute2x128_si256( s0,  s4, 0x20);                        \
  x8  = _mm256_permute2x128_si256( s0,  s4, 0x31);                        \
  x1  = _mm256_permute2x128_si256( s8, s12, 0x20);                        \
  x9  = _mm256_permute2x128_si256( s8, s12, 0x31);                        \
  x2  = _mm256_permute2x128_si256( s1,  s5, 0x20);                        \
  x10 = _mm256_permute2x128_si256( s1,  s5, 0x31);                        \
  x3  = _mm256_permute2x128_si256( s9, s13, 0x20);                        \
  x11 = _mm256_permute2x128_si256( s9, s13, 0x31);                        \
  x4  = _mm256_permute2x128_si256( s2,  s6, 0x20);                        \
  x12 = _mm256_permute2x128_si256( s2,  s6, 0x31);                        \
  x5  = _mm256_permute2x128_si256(s10, s14, 0x20);                        \
  x13 = _mm256_permute2x128_si256(s10, s14, 0x31);                        \
  x6  = _mm256_permute2x128_si256( s3,  s7, 0x20);                        \
  x14 = _mm256_permute2x128_si256( s3,  s7, 0x31);                        \
  x7  = _mm256_permute2x128_si256(s11, s15, 0x20);                        \
  x15 = _mm256_permute2x128_si256(s11, s15, 0x31);                        \
} while(0)

#define CHACHA_OUT(out,in,x0,x1, x2, x3, x4, x5, x6, x7,                  \
                          x8,x9,x10,x11,x12,x13,x14,x15)                  \
do                                                                        \
{                                                                         \
  TRANSPOSE(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15);       \
  STOREU(out +   0, XOR( x0, LOADU(in +   0)));                           \
  STOREU(out +  32, XOR( x1, LOADU(in +  32)));                           \
  STOREU(out +  64, XOR( x2, LOADU(in +  64)));                           \
  STOREU(out +  96, XOR( x3, LOADU(in +  96)));                           \
  STOREU(out + 128, XOR( x4, LOADU(in + 128)));                           \
  STOREU(out + 160, XOR( x5, LOADU(in + 160)));                           \
  STOREU(out + 192, XOR( x6, LOADU(in + 192)));                           \
  STOREU(out + 224, XOR( x7, LOADU(in + 224)));                           \
  STOREU(out + 256, XOR( x8, LOADU(in + 256)));                           \
  STOREU(out + 288, XOR( x9, LOADU(in + 288)));                           \
  STOREU(out + 320, XOR(x10, LOADU(in + 320)));                           \
  STOREU(out + 352, XOR(x11, LOADU(in + 352)));                           \
  STOREU(out + 384, XOR(x12, LOADU(in + 384)));                           \
  STOREU(out + 416, XOR(x13, LOADU(in + 416)));                           \
  STOREU(out + 448, XOR(x14, LOADU(in + 448)));                           \
  STOREU(out + 480, XOR(x15, LOADU(in + 480)));                           \
} while(0)



int crypto_stream_xor(unsigned char *out, const unsigned char *in, 
  unsigned long long inlen, const unsigned char *n_, const unsigned char *k_)
{
  __m256i x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;
  __m256i z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15;
  uint32_t k[8];
  uint32_t n[2];
  
  memcpy(k, k_, 32);
  memcpy(n, n_,  8);

  x0  = _mm256_set1_epi32(0x61707865);
  x1  = _mm256_set1_epi32(0x3320646e);
  x2  = _mm256_set1_epi32(0x79622d32);
  x3  = _mm256_set1_epi32(0x6b206574);
 
  x4  = _mm256_set1_epi32(k[0]);
  x5  = _mm256_set1_epi32(k[1]);
  x6  = _mm256_set1_epi32(k[2]);
  x7  = _mm256_set1_epi32(k[3]);
  x8  = _mm256_set1_epi32(k[4]);
  x9  = _mm256_set1_epi32(k[5]);
  x10 = _mm256_set1_epi32(k[6]);
  x11 = _mm256_set1_epi32(k[7]);

  x12 = _mm256_set_epi32(7,6,5,4,3,2,1,0);
  x13 = _mm256_setzero_si256();
  x14 = _mm256_set1_epi32(n[0]);
  x15 = _mm256_set1_epi32(n[1]);

  while(inlen >= 512)
  {
    CHACHA_CORE(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,\
                x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15);
    CHACHA_OUT(out,in,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15);

    x13 = SUB(x13, _mm256_cmpeq_epi32(x12, _mm256_set_epi32(-1,-2,-3,-4,-5,-6,-7,-8)));
    x12 = ADD(x12, _mm256_set1_epi32(8)); 
    
    out += 512; in += 512; inlen -= 512;
  }

  if(inlen > 0) /* Should fallback to latency-oriented implementation here */
  {
    unsigned char blk[512];
    memcpy(blk, in, inlen);
    CHACHA_CORE(z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,\
                x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15);
    CHACHA_OUT(blk,blk,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15);
    memcpy(out, blk, inlen);
  }

  return 0;
}

int crypto_stream(unsigned char *out, unsigned long long outlen, 
  const unsigned char *n, const unsigned char *k)
{
  memset(out, 0, outlen);
  return crypto_stream_xor(out, out, outlen, n, k);
}

