#include "emprise/backend.hpp"
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace emprise {
namespace {
struct Library {
    void* handle = nullptr;
    explicit Library(const char* name) {
#ifdef _WIN32
        handle = LoadLibraryA(name);
#else
        handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle) throw std::runtime_error(std::string("Cannot load ") + name);
    }
    ~Library() {
#ifdef _WIN32
        if (handle) FreeLibrary(static_cast<HMODULE>(handle));
#else
        if (handle) dlclose(handle);
#endif
    }
    template<class T> T symbol(const char* name) {
#ifdef _WIN32
        auto p = GetProcAddress(static_cast<HMODULE>(handle), name);
#else
        auto p = dlsym(handle, name);
#endif
        if (!p) throw std::runtime_error(std::string("Missing runtime symbol: ") + name);
        return reinterpret_cast<T>(p);
    }
};
using Ptr = uint64_t;
void check(int code, const char* operation) {
    if (code) throw std::runtime_error(std::string(operation) + " failed with code " + std::to_string(code));
}
const char* source = R"CUDA(
__device__ float half_value(unsigned short h) {
    unsigned s = (unsigned)(h & 32768) << 16, e = (h >> 10) & 31, f = h & 1023;
    if (e == 0) return (s ? -1.0f : 1.0f) * (float)f * 5.9604644775390625e-8f;
    return __uint_as_float(s | (e == 31 ? 0x7f800000u : (e + 112) << 23) | (f << 13));
}
__device__ __forceinline__ float half_to_float_fast(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
extern "C" __global__ void linear(const unsigned char* w, const float* x, float* y, int cols, int rows, int type) {
    int row = blockIdx.x, lane = threadIdx.x;
    if (row >= rows) return;
    float sum = 0;
    for (int k = lane; k < cols; k += blockDim.x) {
        float v;
        if (type == 14) {
            const unsigned char* b = w + ((unsigned long long)row * (cols / 256) + k / 256) * 216;
            int i = k % 256, h = i / 128, g = (i % 128) / 32, l = i % 32;
            int lo = (b[h * 64 + (g & 1) * 32 + l] >> (g >= 2 ? 4 : 0)) & 15;
            int hi = (b[128 + h * 32 + l] >> (g * 2)) & 3;
            int q = (lo | (hi << 4)) - 32;
            int sc = ((const signed char*)(b + 192))[h * 8 + g * 2 + l / 16];
            v = half_value(*(const unsigned short*)(b+208)) * sc * q;
        } else if (type == 0) v = ((const float*)w)[(unsigned long long)row * cols + k];
        else {
            unsigned short b = ((const unsigned short*)w)[(unsigned long long)row * cols + k];
            v = type == 30 ? __uint_as_float((unsigned)b << 16) : half_value(b);
        }
        sum += v * x[k];
    }
    __shared__ float partial[128];
    partial[lane] = sum;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) y[row] = partial[0];
}
// Exact Q6_K matrix-vector product with FP32 activations: weights are decoded
// to float (no activation quantization), using the same padded 216-byte VRAM
// layout, hoisted bit positions, and eight weights per thread as linear_q8.
extern "C" __global__ void linear_q6_f32(const unsigned char* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int cols, int rows) {
    int row=blockIdx.x,lane=threadIdx.x;
    if(row>=rows) return;
    int i0=(lane*8)&255, h=i0>>7, g=(i0>>5)&3, l=i0&31;
    int loff=h*64+(g&1)*32+l, hoff=128+h*32+l;
    int shl=(g>=2)?4:0, shh=g*2, sci=h*8+g*2+(l>>4);
    const unsigned long long rowbase=(unsigned long long)row*(cols>>8)*216ull;
    const int nblocks=cols>>8, step=(blockDim.x*8)>>8;
    int block=(lane*8)>>8;
    const float* xp=x+lane*8;
    float sum=0;
    for(;block<nblocks;block+=step,xp+=blockDim.x*8) {
        const unsigned char* b=w+rowbase+(unsigned long long)block*216ull;
        const uint2 lo=*(const uint2*)(b+loff), hi=*(const uint2*)(b+hoff);
        unsigned lo0=lo.x, lo1=lo.y, hi0=hi.x, hi1=hi.y;
        unsigned raw0=((lo0>>shl)&0x0f0f0f0f)|(((hi0>>shh)&0x03030303)<<4);
        unsigned raw1=((lo1>>shl)&0x0f0f0f0f)|(((hi1>>shh)&0x03030303)<<4);
        const float scale=half_to_float_fast(*(const unsigned short*)(b+208))*((const signed char*)(b+192))[sci];
        const float4 xa=*(const float4*)xp, xb=*(const float4*)(xp+4);
        float acc=0;
        acc+=(float)((int)(raw0&0xff)-32)*xa.x;
        acc+=(float)((int)((raw0>>8)&0xff)-32)*xa.y;
        acc+=(float)((int)((raw0>>16)&0xff)-32)*xa.z;
        acc+=(float)((int)((raw0>>24)&0xff)-32)*xa.w;
        acc+=(float)((int)(raw1&0xff)-32)*xb.x;
        acc+=(float)((int)((raw1>>8)&0xff)-32)*xb.y;
        acc+=(float)((int)((raw1>>16)&0xff)-32)*xb.z;
        acc+=(float)((int)((raw1>>24)&0xff)-32)*xb.w;
        sum+=scale*acc;
    }
    __shared__ float partial[128];partial[lane]=sum;__syncthreads();
    for(int stride=64;stride;stride>>=1) { if(lane<stride) partial[lane]+=partial[lane+stride];__syncthreads(); }
    if(!lane) y[row]=partial[0];
}
)CUDA" R"CUDAY(
// Two output rows per thread, reusing the activation loads. The activation
// vector is otherwise re-read from L2 once per output row, which dominates
// traffic for the large projection matrices.
extern "C" __global__ void linear_q6_f32_x2(const unsigned char* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int cols, int rows) {
    int row=blockIdx.x*2,lane=threadIdx.x;
    if(row>=rows) return;
    int i0=(lane*8)&255, h=i0>>7, g=(i0>>5)&3, l=i0&31;
    int loff=h*64+(g&1)*32+l, hoff=128+h*32+l;
    int shl=(g>=2)?4:0, shh=g*2, sci=h*8+g*2+(l>>4);
    const int nblocks=cols>>8, step=(blockDim.x*8)>>8;
    int block=(lane*8)>>8;
    const float* xp=x+lane*8;
    const bool two=row+1<rows;
    float sum0=0,sum1=0;
    for(;block<nblocks;block+=step,xp+=blockDim.x*8) {
        const float4 xa=*(const float4*)xp, xb=*(const float4*)(xp+4);
        const unsigned char* b0=w+((unsigned long long)row*(cols>>8)+block)*216ull;
        const uint2 lo0=*(const uint2*)(b0+loff), hi0=*(const uint2*)(b0+hoff);
        unsigned r0=((lo0.x>>shl)&0x0f0f0f0f)|(((hi0.x>>shh)&0x03030303)<<4);
        unsigned r1=((lo0.y>>shl)&0x0f0f0f0f)|(((hi0.y>>shh)&0x03030303)<<4);
        float s0=half_to_float_fast(*(const unsigned short*)(b0+208))*((const signed char*)(b0+192))[sci];
        float a0=0;
        a0+=(float)((int)(r0&0xff)-32)*xa.x;
        a0+=(float)((int)((r0>>8)&0xff)-32)*xa.y;
        a0+=(float)((int)((r0>>16)&0xff)-32)*xa.z;
        a0+=(float)((int)((r0>>24)&0xff)-32)*xa.w;
        a0+=(float)((int)(r1&0xff)-32)*xb.x;
        a0+=(float)((int)((r1>>8)&0xff)-32)*xb.y;
        a0+=(float)((int)((r1>>16)&0xff)-32)*xb.z;
        a0+=(float)((int)((r1>>24)&0xff)-32)*xb.w;
        sum0+=s0*a0;
        if(two) {
            const unsigned char* b1=w+((unsigned long long)(row+1)*(cols>>8)+block)*216ull;
            const uint2 lo1=*(const uint2*)(b1+loff), hi1=*(const uint2*)(b1+hoff);
            unsigned q0=((lo1.x>>shl)&0x0f0f0f0f)|(((hi1.x>>shh)&0x03030303)<<4);
            unsigned q1=((lo1.y>>shl)&0x0f0f0f0f)|(((hi1.y>>shh)&0x03030303)<<4);
            float s1=half_to_float_fast(*(const unsigned short*)(b1+208))*((const signed char*)(b1+192))[sci];
            float a1=0;
            a1+=(float)((int)(q0&0xff)-32)*xa.x;
            a1+=(float)((int)((q0>>8)&0xff)-32)*xa.y;
            a1+=(float)((int)((q0>>16)&0xff)-32)*xa.z;
            a1+=(float)((int)((q0>>24)&0xff)-32)*xa.w;
            a1+=(float)((int)(q1&0xff)-32)*xb.x;
            a1+=(float)((int)((q1>>8)&0xff)-32)*xb.y;
            a1+=(float)((int)((q1>>16)&0xff)-32)*xb.z;
            a1+=(float)((int)((q1>>24)&0xff)-32)*xb.w;
            sum1+=s1*a1;
        }
    }
    __shared__ float p0[128],p1[128];
    p0[lane]=sum0; if(two) p1[lane]=sum1; __syncthreads();
    for(int stride=64;stride;stride>>=1) {
        if(lane<stride) { p0[lane]+=p0[lane+stride]; if(two) p1[lane]+=p1[lane+stride]; }
        __syncthreads();
    }
    if(!lane) { y[row]=p0[0]; if(two) y[row+1]=p1[0]; }
}
// Four output rows per thread, reusing the activation loads across four rows.
#define Q6_ROW(B,S) { const uint2 lo=*(const uint2*)((B)+loff), hi=*(const uint2*)((B)+hoff); \
    unsigned rr0=((lo.x>>shl)&0x0f0f0f0f)|(((hi.x>>shh)&0x03030303)<<4); \
    unsigned rr1=((lo.y>>shl)&0x0f0f0f0f)|(((hi.y>>shh)&0x03030303)<<4); \
    float sc=half_to_float_fast(*(const unsigned short*)((B)+208))*((const signed char*)((B)+192))[sci]; \
    float aa=0; \
    aa+=(float)((int)(rr0&0xff)-32)*xa.x; aa+=(float)((int)((rr0>>8)&0xff)-32)*xa.y; \
    aa+=(float)((int)((rr0>>16)&0xff)-32)*xa.z; aa+=(float)((int)((rr0>>24)&0xff)-32)*xa.w; \
    aa+=(float)((int)(rr1&0xff)-32)*xb.x; aa+=(float)((int)((rr1>>8)&0xff)-32)*xb.y; \
    aa+=(float)((int)((rr1>>16)&0xff)-32)*xb.z; aa+=(float)((int)((rr1>>24)&0xff)-32)*xb.w; \
    (S)+=sc*aa; }
extern "C" __global__ void linear_q6_f32_x4(const unsigned char* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int cols, int rows) {
    int row=blockIdx.x*4,lane=threadIdx.x;
    if(row>=rows) return;
    int i0=(lane*8)&255, h=i0>>7, g=(i0>>5)&3, l=i0&31;
    int loff=h*64+(g&1)*32+l, hoff=128+h*32+l;
    int shl=(g>=2)?4:0, shh=g*2, sci=h*8+g*2+(l>>4);
    const unsigned long long rs=(cols>>8)*216ull;
    const int nblocks=cols>>8, step=(blockDim.x*8)>>8;
    int block=(lane*8)>>8;
    const float* xp=x+lane*8;
    const int avail=rows-row<4?rows-row:4;
    float s0=0,s1=0,s2=0,s3=0;
    for(;block<nblocks;block+=step,xp+=blockDim.x*8) {
        const float4 xa=*(const float4*)xp, xb=*(const float4*)(xp+4);
        const unsigned char* base=w+rs*row+block*216ull;
        Q6_ROW(base,s0)
        if(avail>1) Q6_ROW(base+rs,s1)
        if(avail>2) Q6_ROW(base+2*rs,s2)
        if(avail>3) Q6_ROW(base+3*rs,s3)
    }
    __shared__ float pr[4][128];
    pr[0][lane]=s0; if(avail>1) pr[1][lane]=s1; if(avail>2) pr[2][lane]=s2; if(avail>3) pr[3][lane]=s3;
    __syncthreads();
    for(int stride=64;stride;stride>>=1) {
        if(lane<stride) {
            pr[0][lane]+=pr[0][lane+stride];
            if(avail>1) pr[1][lane]+=pr[1][lane+stride];
            if(avail>2) pr[2][lane]+=pr[2][lane+stride];
            if(avail>3) pr[3][lane]+=pr[3][lane+stride];
        }
        __syncthreads();
    }
    if(!lane) { y[row]=pr[0][0]; if(avail>1) y[row+1]=pr[1][0]; if(avail>2) y[row+2]=pr[2][0]; if(avail>3) y[row+3]=pr[3][0]; }
}
#undef Q6_ROW
__device__ __forceinline__ float q6_row(const unsigned char* __restrict__ b,int loff,int hoff,int shl,int shh,int sci,const float4& xa,const float4& xb) {
    const uint2 lo=*(const uint2*)(b+loff), hi=*(const uint2*)(b+hoff);
    unsigned a=((lo.x>>shl)&0x0f0f0f0f)|(((hi.x>>shh)&0x03030303)<<4);
    unsigned c=((lo.y>>shl)&0x0f0f0f0f)|(((hi.y>>shh)&0x03030303)<<4);
    const float sc=half_to_float_fast(*(const unsigned short*)(b+208))*((const signed char*)(b+192))[sci];
    float aa=0;
    aa+=(float)((int)(a&0xff)-32)*xa.x;
    aa+=(float)((int)((a>>8)&0xff)-32)*xa.y;
    aa+=(float)((int)((a>>16)&0xff)-32)*xa.z;
    aa+=(float)((int)((a>>24)&0xff)-32)*xa.w;
    aa+=(float)((int)(c&0xff)-32)*xb.x;
    aa+=(float)((int)((c>>8)&0xff)-32)*xb.y;
    aa+=(float)((int)((c>>16)&0xff)-32)*xb.z;
    aa+=(float)((int)((c>>24)&0xff)-32)*xb.w;
    return sc*aa;
}
extern "C" __global__ void linear_q6_f32_x8(const unsigned char* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int cols, int rows) {
    int row=blockIdx.x*8,lane=threadIdx.x;
    if(row>=rows) return;
    int i0=(lane*8)&255, h=i0>>7, g=(i0>>5)&3, l=i0&31;
    int loff=h*64+(g&1)*32+l, hoff=128+h*32+l;
    int shl=(g>=2)?4:0, shh=g*2, sci=h*8+g*2+(l>>4);
    const unsigned long long rs=(cols>>8)*216ull;
    const int nblocks=cols>>8, step=(blockDim.x*8)>>8;
    int block=(lane*8)>>8;
    const float* xp=x+lane*8;
    const bool b1=row+1<rows,b2=row+2<rows,b3=row+3<rows,b4=row+4<rows,b5=row+5<rows,b6=row+6<rows,b7=row+7<rows;
    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    for(;block<nblocks;block+=step,xp+=blockDim.x*8) {
        const float4 xa=*(const float4*)xp, xb=*(const float4*)(xp+4);
        const unsigned char* base=w+rs*row+(unsigned long long)block*216ull;
        s0+=q6_row(base,loff,hoff,shl,shh,sci,xa,xb);
        if(b1) s1+=q6_row(base+rs,loff,hoff,shl,shh,sci,xa,xb);
        if(b2) s2+=q6_row(base+2*rs,loff,hoff,shl,shh,sci,xa,xb);
        if(b3) s3+=q6_row(base+3*rs,loff,hoff,shl,shh,sci,xa,xb);
        if(b4) s4+=q6_row(base+4*rs,loff,hoff,shl,shh,sci,xa,xb);
        if(b5) s5+=q6_row(base+5*rs,loff,hoff,shl,shh,sci,xa,xb);
        if(b6) s6+=q6_row(base+6*rs,loff,hoff,shl,shh,sci,xa,xb);
        if(b7) s7+=q6_row(base+7*rs,loff,hoff,shl,shh,sci,xa,xb);
    }
    __shared__ float pr[8][128];
    pr[0][lane]=s0; if(b1)pr[1][lane]=s1; if(b2)pr[2][lane]=s2; if(b3)pr[3][lane]=s3;
    if(b4)pr[4][lane]=s4; if(b5)pr[5][lane]=s5; if(b6)pr[6][lane]=s6; if(b7)pr[7][lane]=s7;
    __syncthreads();
    for(int stride=64;stride;stride>>=1) {
        if(lane<stride) {
            pr[0][lane]+=pr[0][lane+stride];
            if(b1)pr[1][lane]+=pr[1][lane+stride]; if(b2)pr[2][lane]+=pr[2][lane+stride]; if(b3)pr[3][lane]+=pr[3][lane+stride];
            if(b4)pr[4][lane]+=pr[4][lane+stride]; if(b5)pr[5][lane]+=pr[5][lane+stride]; if(b6)pr[6][lane]+=pr[6][lane+stride]; if(b7)pr[7][lane]+=pr[7][lane+stride];
        }
        __syncthreads();
    }
    if(!lane) {
        y[row]=pr[0][0];
        if(b1)y[row+1]=pr[1][0]; if(b2)y[row+2]=pr[2][0]; if(b3)y[row+3]=pr[3][0];
        if(b4)y[row+4]=pr[4][0]; if(b5)y[row+5]=pr[5][0]; if(b6)y[row+6]=pr[6][0]; if(b7)y[row+7]=pr[7][0];
    }
}
// Batched exact Q6_K: y_j = W * x_j for j in [0, tokens). Each weight group is
// loaded once and reused for every token, and each thread handles two output
// rows so the activation loads are shared across rows as well.
extern "C" __global__ void linear_q6_f32_batch(const unsigned char* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int cols, int rows, int tokens) {
    int row=blockIdx.x*2,lane=threadIdx.x;
    if(row>=rows) return;
    int i0=(lane*8)&255, h=i0>>7, g=(i0>>5)&3, l=i0&31;
    int loff=h*64+(g&1)*32+l, hoff=128+h*32+l;
    int shl=(g>=2)?4:0, shh=g*2, sci=h*8+g*2+(l>>4);
    const unsigned long long rs=(cols>>8)*216ull;
    const int nblocks=cols>>8, step=(blockDim.x*8)>>8;
    const bool two=row+1<rows;
    float acc[4][2];
    #pragma unroll
    for(int j=0;j<4;++j) { acc[j][0]=0; acc[j][1]=0; }
    int xoff=lane*8;
    for(int block=(lane*8)>>8;block<nblocks;block+=step,xoff+=blockDim.x*8) {
        const unsigned char* b0=w+rs*row+(unsigned long long)block*216ull;
        const uint2 lo0=*(const uint2*)(b0+loff), hi0=*(const uint2*)(b0+hoff);
        unsigned r0=((lo0.x>>shl)&0x0f0f0f0f)|(((hi0.x>>shh)&0x03030303)<<4);
        unsigned r1=((lo0.y>>shl)&0x0f0f0f0f)|(((hi0.y>>shh)&0x03030303)<<4);
        const float sc0=half_to_float_fast(*(const unsigned short*)(b0+208))*((const signed char*)(b0+192))[sci];
        const float q0=(float)((int)(r0&0xff)-32), q1=(float)((int)((r0>>8)&0xff)-32);
        const float q2=(float)((int)((r0>>16)&0xff)-32), q3=(float)((int)((r0>>24)&0xff)-32);
        const float q4=(float)((int)(r1&0xff)-32), q5=(float)((int)((r1>>8)&0xff)-32);
        const float q6=(float)((int)((r1>>16)&0xff)-32), q7=(float)((int)((r1>>24)&0xff)-32);
        float p0=0,p1=0,p2=0,p3=0,p4=0,p5=0,p6=0,p7=0,sc1=0;
        if(two) {
            const unsigned char* b1=w+rs*(row+1)+(unsigned long long)block*216ull;
            const uint2 lo1=*(const uint2*)(b1+loff), hi1=*(const uint2*)(b1+hoff);
            unsigned t0=((lo1.x>>shl)&0x0f0f0f0f)|(((hi1.x>>shh)&0x03030303)<<4);
            unsigned t1=((lo1.y>>shl)&0x0f0f0f0f)|(((hi1.y>>shh)&0x03030303)<<4);
            sc1=half_to_float_fast(*(const unsigned short*)(b1+208))*((const signed char*)(b1+192))[sci];
            p0=(float)((int)(t0&0xff)-32); p1=(float)((int)((t0>>8)&0xff)-32);
            p2=(float)((int)((t0>>16)&0xff)-32); p3=(float)((int)((t0>>24)&0xff)-32);
            p4=(float)((int)(t1&0xff)-32); p5=(float)((int)((t1>>8)&0xff)-32);
            p6=(float)((int)((t1>>16)&0xff)-32); p7=(float)((int)((t1>>24)&0xff)-32);
        }
        #pragma unroll
        for(int j=0;j<4;++j) {
            if(j>=tokens) break;
            const float* xk=x+xoff+j*cols;
            const float4 xa=*(const float4*)xk, xb=*(const float4*)(xk+4);
            float d0=q0*xa.x; d0+=q1*xa.y; d0+=q2*xa.z; d0+=q3*xa.w;
            d0+=q4*xb.x; d0+=q5*xb.y; d0+=q6*xb.z; d0+=q7*xb.w;
            acc[j][0]+=sc0*d0;
            if(two) {
                float d1=p0*xa.x; d1+=p1*xa.y; d1+=p2*xa.z; d1+=p3*xa.w;
                d1+=p4*xb.x; d1+=p5*xb.y; d1+=p6*xb.z; d1+=p7*xb.w;
                acc[j][1]+=sc1*d1;
            }
        }
    }
    __shared__ float partial[128];
    #pragma unroll
    for(int j=0;j<4;++j) {
        if(j>=tokens) break;
        partial[lane]=acc[j][0];__syncthreads();
        for(int stride=64;stride;stride>>=1) { if(lane<stride) partial[lane]+=partial[lane+stride];__syncthreads(); }
        if(!lane) y[j*rows+row]=partial[0];
        if(two) {
            partial[lane]=acc[j][1];__syncthreads();
            for(int stride=64;stride;stride>>=1) { if(lane<stride) partial[lane]+=partial[lane+stride];__syncthreads(); }
            if(!lane) y[j*rows+row+1]=partial[0];
        }
    }
}
__device__ __forceinline__ float d_sigmoid(float x){return x>=0?1.0f/(1.0f+expf(-x)):expf(x)/(1.0f+expf(x));}
__device__ __forceinline__ float d_silu(float x){return x*d_sigmoid(x);}
__device__ __forceinline__ float d_softplus(float x){return fmaxf(x,0.0f)+log1pf(expf(-fabsf(x)));}
extern "C" __global__ void delta_conv(float* qkv,const float* conv,float* hist,int channels,int conv_width){
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=channels) return;
    float* h=hist+(size_t)c*conv_width;
    for(int j=0;j<conv_width-1;++j) h[j]=h[j+1];
    h[conv_width-1]=qkv[c];
    float sum=0;
    for(int j=0;j<conv_width;++j) sum+=h[j]*conv[c*conv_width+j];
    qkv[c]=d_silu(sum);
}
extern "C" __global__ void delta_qk_scale(float* qkv,int key_dim,int state_dim,int key_heads,float eps){
    int h=blockIdx.x;
    if(h>=key_heads) return;
    float* q=qkv+(size_t)h*state_dim;
    float* k=qkv+key_dim+(size_t)h*state_dim;
    __shared__ float sq[128],sk[128];
    float qs=0,ks=0;
    for(int d=threadIdx.x;d<state_dim;d+=blockDim.x){ qs+=q[d]*q[d]; ks+=k[d]*k[d]; }
    sq[threadIdx.x]=qs; sk[threadIdx.x]=ks; __syncthreads();
    for(int stride=blockDim.x>>1;stride;stride>>=1){ if(threadIdx.x<stride){sq[threadIdx.x]+=sq[threadIdx.x+stride];sk[threadIdx.x]+=sk[threadIdx.x+stride];} __syncthreads(); }
    if(threadIdx.x==0){ sq[0]=1.0f/sqrtf(sq[0]+eps)/sqrtf((float)state_dim); sk[0]=1.0f/sqrtf(sk[0]+eps); }
    __syncthreads();
    float qscale=sq[0],kscale=sk[0];
    for(int d=threadIdx.x;d<state_dim;d+=blockDim.x){ q[d]*=qscale; k[d]*=kscale; }
}
extern "C" __global__ void delta_scan(const float* qkv,const float* alpha,const float* beta,const float* ssm_a,const float* dt,float* state,float* out,int key_dim,int state_dim,int key_heads,int value_heads,float eps){
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int hd=blockIdx.x*(blockDim.x>>5)+warp;
    if(hd>=value_heads*state_dim) return;
    int h=hd/state_dim,d=hd%state_dim,kh=h%key_heads;
    const float* q=qkv+(size_t)kh*state_dim;
    const float* k=qkv+key_dim+(size_t)kh*state_dim;
    const float* v=qkv+2*key_dim+(size_t)h*state_dim;
    float* s=state+((size_t)h*state_dim+d)*state_dim;
    // query/key L2 scaling, folded in so no separate kernel is needed.
    float qn=0,kn=0;
    for(int j=lane;j<state_dim;j+=32){ qn+=q[j]*q[j]; kn+=k[j]*k[j]; }
    for(int off=16;off;off>>=1){ qn+=__shfl_down_sync(0xffffffffu,qn,off); kn+=__shfl_down_sync(0xffffffffu,kn,off); }
    const float qscale=1.0f/sqrtf(__shfl_sync(0xffffffffu,qn,0)+eps)/sqrtf((float)state_dim);
    const float kscale=1.0f/sqrtf(__shfl_sync(0xffffffffu,kn,0)+eps);
    float decay=expf(ssm_a[h]*d_softplus(alpha[h]+dt[h]));
    float b=d_sigmoid(beta[h]);
    float mem=0;
    for(int j=lane;j<state_dim;j+=32){ s[j]*=decay; mem+=s[j]*k[j]*kscale; }
    for(int off=16;off;off>>=1) mem+=__shfl_down_sync(0xffffffffu,mem,off);
    float delta=(v[d]-__shfl_sync(0xffffffffu,mem,0))*b;
    float sm=0;
    for(int j=lane;j<state_dim;j+=32){ s[j]+=k[j]*kscale*delta; sm+=s[j]*q[j]*qscale; }
    for(int off=16;off;off>>=1) sm+=__shfl_down_sync(0xffffffffu,sm,off);
    if(!lane) out[h*state_dim+d]=sm;
}
extern "C" __global__ void delta_norm_gate(float* out,const float* gate,const float* nw,float eps,int state_dim){
    int h=blockIdx.x;
    float* o=out+(size_t)h*state_dim;
    const float* g=gate+(size_t)h*state_dim;
    __shared__ float red[128];
    float s=0;
    for(int d=threadIdx.x;d<state_dim;d+=blockDim.x) s+=o[d]*o[d];
    red[threadIdx.x]=s; __syncthreads();
    for(int stride=blockDim.x>>1;stride;stride>>=1){ if(threadIdx.x<stride) red[threadIdx.x]+=red[threadIdx.x+stride]; __syncthreads(); }
    float scale=1.0f/sqrtf(red[0]/state_dim+eps);
    for(int d=threadIdx.x;d<state_dim;d+=blockDim.x) o[d]=o[d]*scale*nw[d]*d_silu(g[d]);
}
)CUDAY" R"CUDAX(
extern "C" __global__ void quantize_x(const float* x, signed char* q, float* scales, int cols) {
    int lane=threadIdx.x, i=blockIdx.x*32+lane;
    float v=i<cols?x[i]:0.0f, a=fabsf(v);
    for(int d=16;d;d>>=1) a=fmaxf(a,__shfl_down_sync(0xffffffff,a,d));
    a=__shfl_sync(0xffffffff,a,0);
    float scale=a/127.0f;
    // Q8_1 stores its scale as FP16; quantization itself uses the unrounded
    // FP32 scale and round-to-nearest with ties away from zero.
    unsigned short packed_scale;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(packed_scale) : "f"(scale));
    if(lane==0) scales[blockIdx.x]=half_value(packed_scale);
    if(i<cols) q[i]=(signed char)(scale?roundf(v/scale):0);
}
extern "C" __global__ void swiglu(float* gate, const float* up, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {
        float g=gate[i], e=expf(-fabsf(g));
        gate[i]=g*(g>=0?1.0f/(1.0f+e):e/(1.0f+e))*up[i];
    }
}
extern "C" __global__ void linear_q8(const unsigned char* __restrict__ w, const signed char* __restrict__ x, const float* __restrict__ scales, float* __restrict__ y, int cols, int rows) {
    int row=blockIdx.x,lane=threadIdx.x;
    if(row>=rows) return;
    // Eight weights per thread; the stride is a multiple of 256 so the bit
    // positions, sub-block scale index, and super-scale stay loop-invariant.
    int i0=(lane*8)&255, h=i0>>7, g=(i0>>5)&3, l=i0&31;
    int loff=h*64+(g&1)*32+l, hoff=128+h*32+l;
    int shl=(g>=2)?4:0, shh=g*2, sci=h*8+g*2+(l>>4);
    const unsigned long long rowbase=(unsigned long long)row*(cols>>8)*216ull;
    const int nblocks=cols>>8, step=(blockDim.x*8)>>8;
    int block=(lane*8)>>8;
    const signed char* xp=x+lane*8;
    float sum=0;
    for(;block<nblocks;block+=step,xp+=blockDim.x*8) {
        const unsigned char* b=w+rowbase+(unsigned long long)block*216ull;
        const uint2 lo=*(const uint2*)(b+loff), hi=*(const uint2*)(b+hoff);
        unsigned lo0=lo.x, lo1=lo.y, hi0=hi.x, hi1=hi.y;
        unsigned raw0=((lo0>>shl)&0x0f0f0f0f)|(((hi0>>shh)&0x03030303)<<4);
        unsigned raw1=((lo1>>shl)&0x0f0f0f0f)|(((hi1>>shh)&0x03030303)<<4);
        unsigned sq0=((raw0|0x80808080)-0x20202020)^0x80808080;
        unsigned sq1=((raw1|0x80808080)-0x20202020)^0x80808080;
        int dot=__dp4a((int)sq0,*(const int*)xp,0);
        dot=__dp4a((int)sq1,*(const int*)(xp+4),dot);
        int sc=((const signed char*)(b+192))[sci];
        sum+=(float)dot*(half_to_float_fast(*(const unsigned short*)(b+208))*sc)*scales[(xp-x)>>5];
    }
    __shared__ float partial[128];partial[lane]=sum;__syncthreads();
    for(int stride=64;stride;stride>>=1) { if(lane<stride) partial[lane]+=partial[lane+stride];__syncthreads(); }
    if(!lane) y[row]=partial[0];
}
)CUDAX";

class Cuda final: public Backend {
#ifdef _WIN32
    Library driver{"nvcuda.dll"};
    Library rtc{std::getenv("EMPRISE_NVRTC_LIBRARY") ? std::getenv("EMPRISE_NVRTC_LIBRARY") : "nvrtc64_120_0.dll"};
#else
    Library driver{"libcuda.so.1"};
    Library rtc{std::getenv("EMPRISE_NVRTC_LIBRARY") ? std::getenv("EMPRISE_NVRTC_LIBRARY") : "libnvrtc.so.12"};
#endif
    int (*alloc)(Ptr*, size_t) = driver.symbol<decltype(alloc)>("cuMemAlloc_v2");
    int (*free_mem)(Ptr) = driver.symbol<decltype(free_mem)>("cuMemFree_v2");
    int (*htod)(Ptr, const void*, size_t) = driver.symbol<decltype(htod)>("cuMemcpyHtoD_v2");
    int (*dtoh)(void*, Ptr, size_t) = driver.symbol<decltype(dtoh)>("cuMemcpyDtoH_v2");
    int (*set_context)(void*) = driver.symbol<decltype(set_context)>("cuCtxSetCurrent");
    int (*launch)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**) = driver.symbol<decltype(launch)>("cuLaunchKernel");
    void *context = nullptr, *module = nullptr, *kernel = nullptr, *q8_kernel = nullptr, *q6_f32_kernel = nullptr, *q6_f32_x2_kernel = nullptr, *q6_f32_x4_kernel = nullptr, *q6_f32_x8_kernel = nullptr, *q6_f32_batch_kernel = nullptr, *quantize_kernel = nullptr, *swiglu_kernel = nullptr;
    void *delta_conv_kernel = nullptr, *delta_qk_kernel = nullptr, *delta_scan_kernel = nullptr, *delta_norm_kernel = nullptr;
    Ptr dn_scratch = 0; size_t dn_scratch_size = 0;
    Ptr rb_proj = 0; size_t rb_proj_size = 0;
    Ptr rb_dn = 0; size_t rb_dn_size = 0;
    struct DnSlot { Ptr conv=0, rec=0, params=0; size_t params_n=0; };
    std::unordered_map<int, DnSlot> dn_slots;
    Ptr input = 0, output = 0, quantized = 0, scales = 0;
    size_t input_size = 0, output_size = 0, quantized_size = 0, scales_size = 0;
    Ptr gate_buffer=0, up_buffer=0;
    size_t gate_size=0, up_size=0;
    bool fast = false;
    uint64_t budget = 0, used = 0;
    std::unordered_map<std::string, Ptr> weights;
    const Gguf* owner = nullptr;
    std::unique_ptr<Backend> cpu = cpu_backend();
    std::string label;
    void buffer(Ptr& p, size_t& capacity, size_t required) {
        if (required <= capacity) return;
        if (p) { check(free_mem(p), "free buffer"); p = 0; capacity = 0; }
        check(alloc(&p, required), "allocate buffer"); capacity = required;
    }
    void bind(Gguf& model) {
        if(owner && owner!=&model) throw std::invalid_argument("CUDA backend is bound to one model instance");
        owner=&model;check(set_context(context),"set context");
    }
    // Q6_K super-blocks are padded from 210 to 216 bytes in VRAM so that every
    // block and every four-weight offset stays 4-byte aligned for word loads.
    static uint64_t vram_bytes(const Tensor& t) {
        return t.type==WeightType::q6_k ? (t.elements/256)*216 : t.bytes;
    }
    Ptr resident(Gguf& model, const Tensor& t) {
        auto found=weights.find(t.name);
        if(found!=weights.end()) return found->second;
        const uint64_t size=vram_bytes(t);
        if(size>budget-used) return 0;
        Ptr ptr=0;check(alloc(&ptr,static_cast<size_t>(size)),"allocate weights");
        try {
            if(t.type==WeightType::q6_k) {
                const uint64_t blocks=t.elements/256;
                const uint64_t per_tile=std::max<uint64_t>(1,(4*1024*1024)/210);
                std::vector<std::byte> in(static_cast<size_t>(per_tile*210)), out(static_cast<size_t>(per_tile*216));
                for(uint64_t done=0;done<blocks;done+=per_tile) {
                    const uint64_t n=std::min(per_tile,blocks-done);
                    model.read(t,done*210,std::span(in).first(static_cast<size_t>(n*210)));
                    for(uint64_t i=0;i<n;++i) {
                        std::memcpy(out.data()+i*216,in.data()+i*210,210);
                        for(int z=210;z<216;++z) out[i*216+z]=std::byte{0};
                    }
                    check(htod(ptr+done*216,out.data(),static_cast<size_t>(n*216)),"upload weights");
                }
            } else {
                std::vector<std::byte> tile(static_cast<size_t>(std::min<uint64_t>(t.bytes,8*1024*1024)));
                for(uint64_t start=0;start<t.bytes;start+=tile.size()) {
                    auto slice=std::span(tile).first(static_cast<size_t>(std::min<uint64_t>(tile.size(),t.bytes-start)));
                    model.read(t,start,slice);check(htod(ptr+start,slice.data(),slice.size()),"upload weights");
                }
            }
            weights.emplace(t.name,ptr);
        } catch(...) {free_mem(ptr);throw;}
        used+=size;return ptr;
    }
    void quantize_input(Ptr x,int cols) {
        buffer(quantized,quantized_size,cols);buffer(scales,scales_size,cols/32*sizeof(float));
        void* args[]={&x,&quantized,&scales,&cols};
        check(launch(quantize_kernel,cols/32,1,1,32,1,1,0,nullptr,args,nullptr),"quantize input");
    }
    void device_linear(const Tensor& t,Ptr w,Ptr x,Ptr y,bool reuse_quantized=false) {
        int cols=static_cast<int>(t.shape[0]),rows=static_cast<int>(t.shape[1]),type=static_cast<int>(t.type);
        if(fast && t.type==WeightType::q6_k) {
            if(!reuse_quantized) quantize_input(x,cols);
            void* args[]={&w,&quantized,&scales,&y,&cols,&rows};
            check(launch(q8_kernel,rows,1,1,128,1,1,0,nullptr,args,nullptr),"launch Q6/Q8 matvec");
        } else if(t.type==WeightType::q6_k) {
            void* args[]={&w,&x,&y,&cols,&rows};
            check(launch(q6_f32_x4_kernel,(rows+3)/4,1,1,128,1,1,0,nullptr,args,nullptr),"launch Q6 FP32 x4 matvec");
        } else {
            void* args[]={&w,&x,&y,&cols,&rows,&type};
            check(launch(kernel,rows,1,1,128,1,1,0,nullptr,args,nullptr),"launch matvec");
        }
    }
public:
    explicit Cuda(uint64_t requested, bool quantize): fast(quantize) {
        check(driver.symbol<int(*)(unsigned)>("cuInit")(0), "cuInit");
        int count = 0;
        check(driver.symbol<int(*)(int*)>("cuDeviceGetCount")(&count), "device count");
        if (!count) throw std::runtime_error("No CUDA device");
        int major = 0, minor = 0;
        auto attribute = driver.symbol<int(*)(int*, int, int)>("cuDeviceGetAttribute");
        check(attribute(&major, 75, 0), "compute capability");
        check(attribute(&minor, 76, 0), "compute capability");
        if (major < 6 || (major == 6 && minor < 1)) throw std::runtime_error("CUDA backend requires compute capability 6.1 or newer");
        char name[256]{};
        check(driver.symbol<int(*)(char*, int, int)>("cuDeviceGetName")(name, 256, 0), "device name");
        check(driver.symbol<int(*)(void**, int)>("cuDevicePrimaryCtxRetain")(&context, 0), "retain context");
        try {
            check(set_context(context), "set context");
            size_t available = 0, total = 0;
            check(driver.symbol<int(*)(size_t*, size_t*)>("cuMemGetInfo_v2")(&available, &total), "memory info");
            constexpr size_t reserve = 512ull * 1024 * 1024;
            if (available <= reserve) throw std::runtime_error("Insufficient free VRAM");
            budget = requested ? std::min<uint64_t>(requested, available - reserve) : available - reserve;
            void* program = nullptr;
            check(rtc.symbol<int(*)(void**, const char*, const char*, int, const char* const*, const char* const*)>("nvrtcCreateProgram")(&program, source, "emprise.cu", 0, nullptr, nullptr), "create CUDA program");
            auto destroy = rtc.symbol<int(*)(void**)>("nvrtcDestroyProgram");
            std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
            const bool verbose = std::getenv("EMPRISE_NVRTC_VERBOSE") != nullptr;
            std::vector<const char*> options = {arch.c_str(), "--std=c++14", "--fmad=true"};
            if (verbose) options.push_back("--ptxas-options=-v");
            auto compiled = rtc.symbol<int(*)(void*, int, const char* const*)>("nvrtcCompileProgram")(program, static_cast<int>(options.size()), options.data());
            size_t size = 0;
            rtc.symbol<int(*)(void*, size_t*)>("nvrtcGetProgramLogSize")(program, &size);
            std::string log(size, '\0');
            rtc.symbol<int(*)(void*, char*)>("nvrtcGetProgramLog")(program, log.data());
            if (verbose) std::cerr << "NVRTC log:\n" << log << std::endl;
            if (compiled) { destroy(&program); throw std::runtime_error("NVRTC compilation failed: " + log); }
            check(rtc.symbol<int(*)(void*, size_t*)>("nvrtcGetPTXSize")(program, &size), "PTX size");
            std::string ptx(size, '\0');
            check(rtc.symbol<int(*)(void*, char*)>("nvrtcGetPTX")(program, ptx.data()), "PTX");
            destroy(&program);
            check(driver.symbol<int(*)(void**, const void*)>("cuModuleLoadData")(&module, ptx.data()), "load kernel module");
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&kernel, module, "linear"), "get kernel");
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&q8_kernel, module, "linear_q8"), "get Q8 kernel");
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&q6_f32_kernel, module, "linear_q6_f32"), "get Q6 FP32 kernel");
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&q6_f32_x2_kernel, module, "linear_q6_f32_x2"), "get Q6 FP32 x2 kernel");
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&q6_f32_x4_kernel, module, "linear_q6_f32_x4"), "get Q6 FP32 x4 kernel");
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&q6_f32_x8_kernel, module, "linear_q6_f32_x8"), "get Q6 FP32 x8 kernel");
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&q6_f32_batch_kernel, module, "linear_q6_f32_batch"), "get Q6 FP32 batch kernel");
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&delta_conv_kernel, module, "delta_conv"), "get delta conv kernel");
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&delta_qk_kernel, module, "delta_qk_scale"), "get delta qk kernel");
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&delta_scan_kernel, module, "delta_scan"), "get delta scan kernel");
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&delta_norm_kernel, module, "delta_norm_gate"), "get delta norm kernel");
            if (verbose) {
                auto getattr = driver.symbol<int(*)(int*, int, void*)>("cuFuncGetAttribute");
                for (auto [name, fn] : {std::pair<const char*, void*>("linear", kernel),
                                         std::pair<const char*, void*>("linear_q8", q8_kernel),
                                         std::pair<const char*, void*>("linear_q6_f32", q6_f32_kernel),
                                         std::pair<const char*, void*>("linear_q6_f32_batch", q6_f32_batch_kernel)}) {
                    int regs = 0, local = 0;
                    getattr(&regs, 4, fn);   // CU_FUNC_ATTRIBUTE_NUM_REGS
                    getattr(&local, 3, fn);  // CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
                    std::cerr << "kernel " << name << " regs=" << regs << " local_bytes=" << local << '\n';
                }
            }
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&quantize_kernel, module, "quantize_x"), "get quantizer");
            check(driver.symbol<int(*)(void**, void*, const char*)>("cuModuleGetFunction")(&swiglu_kernel,module,"swiglu"),"get SwiGLU kernel");
            label = std::string("CUDA ") + name + " compute_" + std::to_string(major) + std::to_string(minor);
            if (fast) label += " Q8 activations (experimental)";
        } catch (...) {
            if (module) driver.symbol<int(*)(void*)>("cuModuleUnload")(module);
            driver.symbol<int(*)(int)>("cuDevicePrimaryCtxRelease_v2")(0);
            throw;
        }
    }
    ~Cuda() override {
        set_context(context);
        for (const auto& [key, ptr] : weights) free_mem(ptr);
        if (input) free_mem(input);
        if (output) free_mem(output);
        if (quantized) free_mem(quantized);
        if (scales) free_mem(scales);
        if(gate_buffer) free_mem(gate_buffer);
        if(up_buffer) free_mem(up_buffer);
        if(dn_scratch) free_mem(dn_scratch);
        if(rb_proj) free_mem(rb_proj);
        if(rb_dn) free_mem(rb_dn);
        for(auto& [slot,st]:dn_slots) { if(st.conv) free_mem(st.conv); if(st.rec) free_mem(st.rec); if(st.params) free_mem(st.params); }
        if (module) driver.symbol<int(*)(void*)>("cuModuleUnload")(module);
        driver.symbol<int(*)(int)>("cuDevicePrimaryCtxRelease_v2")(0);
    }
    std::string name() const override { return label; }
    void linear(Gguf& model, const Tensor& t, std::span<const float> x, std::span<float> y) override {
        bind(model);
        if (t.shape.size() != 2 || t.shape[0] != x.size() || t.shape[1] != y.size() ||
            x.size() > INT_MAX || y.size() > INT_MAX) throw std::invalid_argument("CUDA linear shape mismatch");
        Ptr ptr=resident(model,t);
        if(!ptr) {cpu->linear(model,t,x,y);return;}
        buffer(input, input_size, x.size_bytes()); buffer(output, output_size, y.size_bytes());
        check(htod(input, x.data(), x.size_bytes()), "upload activation");
        device_linear(t,ptr,input,output);
        check(dtoh(y.data(), output, y.size_bytes()), "read activation");
    }
    void linear_many(Gguf& model, std::span<const LinearRequest> requests, std::span<const float> x) override {
        if(requests.empty()) return;
        bind(model);
        size_t total=0; bool all_q6=true;
        for(const auto& r:requests) { total+=r.output.size(); if(r.tensor->type!=WeightType::q6_k) all_q6=false; }
        std::vector<Ptr> ptrs(requests.size());
        for(size_t i=0;i<requests.size();++i) {
            if(!x.size() || requests[i].tensor->shape.size()!=2 || requests[i].tensor->shape[0]!=x.size() ||
               requests[i].tensor->shape[1]!=requests[i].output.size() || x.size()>INT_MAX || total>INT_MAX)
                throw std::invalid_argument("CUDA grouped linear shape mismatch");
            ptrs[i]=resident(model,*requests[i].tensor);
            if(!ptrs[i]) {Backend::linear_many(model,requests,x);return;}
        }
        buffer(input,input_size,x.size_bytes());
        buffer(output,output_size,total*sizeof(float));
        check(htod(input,x.data(),x.size_bytes()),"upload activation");
        const bool reuse=fast&&all_q6;
        if(reuse) quantize_input(input,static_cast<int>(x.size()));
        size_t offset=0;
        for(size_t i=0;i<requests.size();++i) {
            device_linear(*requests[i].tensor,ptrs[i],input,output+offset*sizeof(float),reuse);
            offset+=requests[i].output.size();
        }
        std::vector<float> staging(total);
        check(dtoh(staging.data(),output,total*sizeof(float)),"read activation");
        offset=0;
        for(const auto& r:requests) {
            std::memcpy(r.output.data(),staging.data()+offset,r.output.size()*sizeof(float));
            offset+=r.output.size();
        }
    }
    void linear_batch(Gguf& model, const Tensor& t, std::span<const float> x, std::span<float> y, int K) override {
        if(K<=1 || K>4 || t.type!=WeightType::q6_k || t.shape.size()!=2) {Backend::linear_batch(model,t,x,y,K);return;}
        bind(model);
        const int cols=static_cast<int>(t.shape[0]), rows=static_cast<int>(t.shape[1]);
        if(x.size()!=static_cast<size_t>(cols)*static_cast<size_t>(K) ||
           y.size()!=static_cast<size_t>(rows)*static_cast<size_t>(K) ||
           x.size()>INT_MAX || y.size()>INT_MAX)
            throw std::invalid_argument("CUDA batched linear shape mismatch");
        Ptr w=resident(model,t);
        if(!w) {Backend::linear_batch(model,t,x,y,K);return;}
        buffer(input,input_size,x.size_bytes());
        buffer(output,output_size,y.size_bytes());
        check(htod(input,x.data(),x.size_bytes()),"upload batch");
        int c=cols, r=rows, n=K;
        void* args[]={&w,&input,&output,&c,&r,&n};
        check(launch(q6_f32_batch_kernel,(rows+1)/2,1,1,128,1,1,0,nullptr,args,nullptr),"launch Q6 FP32 batch matvec");
        check(dtoh(y.data(),output,y.size_bytes()),"read batch");
    }
    void feed_forward(Gguf& model,const Tensor& gate,const Tensor& up,const Tensor& down,
                      std::span<const float> x,std::span<float> y) override {
        if(gate.shape.size()!=2 || up.shape!=gate.shape || down.shape.size()!=2 ||
           gate.shape[0]!=x.size() || down.shape[0]!=gate.shape[1] || down.shape[1]!=y.size() ||
           x.size()>INT_MAX || y.size()>INT_MAX || gate.shape[1]>INT_MAX)
            throw std::invalid_argument("CUDA feed-forward shape mismatch");
        bind(model);
        uint64_t remaining=budget-used;
        for(const Tensor* t:{&gate,&up,&down}) if(!weights.contains(t->name)) {
            const uint64_t size=vram_bytes(*t);
            if(size>remaining) {Backend::feed_forward(model,gate,up,down,x,y);return;}
            remaining-=size;
        }
        Ptr gw=resident(model,gate),uw=resident(model,up),dw=resident(model,down);
        int n=static_cast<int>(gate.shape[1]);
        buffer(input,input_size,x.size_bytes());buffer(output,output_size,y.size_bytes());
        buffer(gate_buffer,gate_size,size_t(n)*sizeof(float));buffer(up_buffer,up_size,size_t(n)*sizeof(float));
        // Reserve for both projections before launching; reallocating a scratch
        // buffer mid-chain would force an implicit device synchronization.
        if(fast) {
            size_t widest=std::max(x.size(),size_t(n));
            buffer(quantized,quantized_size,widest);buffer(scales,scales_size,widest/32*sizeof(float));
        }
        check(htod(input,x.data(),x.size_bytes()),"upload feed-forward input");
        device_linear(gate,gw,input,gate_buffer);
        device_linear(up,uw,input,up_buffer,fast && gate.type==WeightType::q6_k && up.type==WeightType::q6_k);
        void* args[]={&gate_buffer,&up_buffer,&n};
        check(launch(swiglu_kernel,(n+127)/128,1,1,128,1,1,0,nullptr,args,nullptr),"launch SwiGLU");
        device_linear(down,dw,gate_buffer,output);
        check(dtoh(y.data(),output,y.size_bytes()),"read feed-forward output");
    }
    void reset_state() override {
        if(!context) return;
        check(set_context(context),"set context");
        for(auto& [slot,st]:dn_slots) { if(st.conv) free_mem(st.conv); if(st.rec) free_mem(st.rec); if(st.params) free_mem(st.params); }
        dn_slots.clear();
    }
    void delta_net(Gguf&, const Backend::DeltaNet& p) override {
        const int key_dim=p.key_heads*p.state_dim, value_dim=p.value_heads*p.state_dim, channels=2*key_dim+value_dim;
        if(p.qkv.size()!=size_t(channels) || p.gate.size()!=size_t(value_dim) ||
           p.conv.size()!=size_t(channels*p.conv_width) || p.ssm_a.size()!=size_t(p.value_heads) ||
           p.dt.size()!=size_t(p.value_heads) || p.norm.size()!=size_t(p.state_dim) || p.out.size()!=size_t(value_dim))
            throw std::invalid_argument("CUDA delta_net shape mismatch");
        check(set_context(context),"set context");
        auto& slot=dn_slots[p.slot];
        if(!slot.conv) {
            check(alloc(&slot.conv,size_t(channels)*p.conv_width*sizeof(float)),"allocate delta conv state");
            check(alloc(&slot.rec,size_t(value_dim)*p.state_dim*sizeof(float)),"allocate delta rec state");
            std::vector<float> zeros(static_cast<size_t>(std::max(size_t(channels)*p.conv_width,size_t(value_dim)*p.state_dim)),0.f);
            check(htod(slot.conv,zeros.data(),size_t(channels)*p.conv_width*sizeof(float)),"zero delta conv state");
            check(htod(slot.rec,zeros.data(),size_t(value_dim)*p.state_dim*sizeof(float)),"zero delta rec state");
        }
        Ptr conv_state=slot.conv, rec_state=slot.rec;
        const size_t qkv_n=size_t(channels), gate_n=size_t(value_dim), ab_n=size_t(p.value_heads);
        const size_t conv_n=size_t(channels)*p.conv_width, norm_n=size_t(p.state_dim);
        if(!slot.params) {
            slot.params_n=conv_n+3*ab_n+norm_n;
            check(alloc(&slot.params,slot.params_n*sizeof(float)),"allocate delta params");
            std::vector<float> packed(slot.params_n);
            std::copy(p.conv.begin(),p.conv.end(),packed.begin());
            std::copy(p.ssm_a.begin(),p.ssm_a.end(),packed.begin()+conv_n);
            std::copy(p.dt.begin(),p.dt.end(),packed.begin()+conv_n+ab_n);
            std::copy(p.norm.begin(),p.norm.end(),packed.begin()+conv_n+2*ab_n);
            check(htod(slot.params,packed.data(),slot.params_n*sizeof(float)),"upload delta params");
        }
        const size_t off_qkv=0, off_gate=off_qkv+qkv_n, off_alpha=off_gate+gate_n, off_beta=off_alpha+ab_n;
        std::vector<float> packed(off_beta+ab_n);
        std::copy(p.qkv.begin(),p.qkv.end(),packed.begin()+off_qkv);
        std::copy(p.gate.begin(),p.gate.end(),packed.begin()+off_gate);
        std::copy(p.alpha.begin(),p.alpha.end(),packed.begin()+off_alpha);
        std::copy(p.beta.begin(),p.beta.end(),packed.begin()+off_beta);
        buffer(dn_scratch,dn_scratch_size,packed.size()*sizeof(float));
        check(htod(dn_scratch,packed.data(),packed.size()*sizeof(float)),"upload delta inputs");
        buffer(output,output_size,size_t(value_dim)*sizeof(float));
        Ptr qkv_p=dn_scratch+off_qkv*sizeof(float), gate_p=dn_scratch+off_gate*sizeof(float);
        Ptr alpha_p=dn_scratch+off_alpha*sizeof(float), beta_p=dn_scratch+off_beta*sizeof(float);
        Ptr conv_p=slot.params, a_p=slot.params+conv_n*sizeof(float);
        Ptr dt_p=a_p+ab_n*sizeof(float), norm_p=dt_p+ab_n*sizeof(float);
        int c=channels, sd=p.state_dim, kh=p.key_heads, vh=p.value_heads, cw=p.conv_width, kd=key_dim;
        float eps=p.eps;
        { void* args[]={&qkv_p,&conv_p,&conv_state,&c,&cw};
          check(launch(delta_conv_kernel,(channels+255)/256,1,1,256,1,1,0,nullptr,args,nullptr),"launch delta conv"); }
        { void* args[]={&qkv_p,&alpha_p,&beta_p,&a_p,&dt_p,&rec_state,&output,&kd,&sd,&kh,&vh,&eps};
          check(launch(delta_scan_kernel,(vh*sd+7)/8,1,1,256,1,1,0,nullptr,args,nullptr),"launch delta scan"); }
        { void* args[]={&output,&gate_p,&norm_p,&eps,&sd};
          check(launch(delta_norm_kernel,vh,1,1,128,1,1,0,nullptr,args,nullptr),"launch delta norm"); }
        check(dtoh(p.out.data(),output,size_t(value_dim)*sizeof(float)),"read delta out");
    }
    void recurrent_block(Gguf& model, const Backend::RecurrentBlock& p) override {
        const int key_dim=p.key_heads*p.state_dim, value_dim=p.value_heads*p.state_dim, channels=2*key_dim+value_dim;
        if(p.x.empty() || p.out.size()!=p.x.size() || p.qkv_w->shape.size()!=2 ||
           p.qkv_w->shape[0]!=p.x.size() || p.gate_w->shape[0]!=p.x.size() ||
           p.out_w->shape.size()!=2 || p.out_w->shape[1]!=p.x.size() ||
           p.conv.size()!=size_t(channels*p.conv_width))
            throw std::invalid_argument("CUDA recurrent_block shape mismatch");
        bind(model);
        Ptr wq=resident(model,*p.qkv_w),wg=resident(model,*p.gate_w),wa=resident(model,*p.alpha_w),
            wb=resident(model,*p.beta_w),wo=resident(model,*p.out_w);
        if(!wq||!wg||!wa||!wb||!wo) {Backend::recurrent_block(model,p);return;}
        auto& slot=dn_slots[p.slot];
        if(!slot.conv) {
            check(alloc(&slot.conv,size_t(channels)*p.conv_width*sizeof(float)),"allocate delta conv state");
            check(alloc(&slot.rec,size_t(value_dim)*p.state_dim*sizeof(float)),"allocate delta rec state");
            std::vector<float> zeros(static_cast<size_t>(std::max(size_t(channels)*p.conv_width,size_t(value_dim)*p.state_dim)),0.f);
            check(htod(slot.conv,zeros.data(),size_t(channels)*p.conv_width*sizeof(float)),"zero delta conv state");
            check(htod(slot.rec,zeros.data(),size_t(value_dim)*p.state_dim*sizeof(float)),"zero delta rec state");
        }
        Ptr conv_state=slot.conv, rec_state=slot.rec;
        buffer(input,input_size,p.x.size_bytes());
        check(htod(input,p.x.data(),p.x.size_bytes()),"upload recurrent input");
        const size_t proj_total=size_t(channels)+size_t(value_dim)+2*size_t(p.value_heads);
        buffer(rb_proj,rb_proj_size,proj_total*sizeof(float));
        Ptr qkv_p=rb_proj, gate_p=rb_proj+size_t(channels)*sizeof(float);
        Ptr alpha_p=gate_p+size_t(value_dim)*sizeof(float), beta_p=alpha_p+size_t(p.value_heads)*sizeof(float);
        const bool fastq=fast;
        device_linear(*p.qkv_w,wq,input,qkv_p);
        device_linear(*p.gate_w,wg,input,gate_p,fastq && p.gate_w->type==WeightType::q6_k);
        device_linear(*p.alpha_w,wa,input,alpha_p,fastq && p.alpha_w->type==WeightType::q6_k);
        device_linear(*p.beta_w,wb,input,beta_p,fastq && p.beta_w->type==WeightType::q6_k);
        // conv | ssm_a | dt | norm parameters are constant per layer: upload once.
        const size_t conv_n=size_t(channels)*p.conv_width, ab_n=size_t(p.value_heads), norm_n=size_t(p.state_dim);
        if(!slot.params) {
            slot.params_n=conv_n+3*ab_n+norm_n;
            check(alloc(&slot.params,slot.params_n*sizeof(float)),"allocate recurrent params");
            std::vector<float> packed(slot.params_n);
            std::copy(p.conv.begin(),p.conv.end(),packed.begin());
            std::copy(p.ssm_a.begin(),p.ssm_a.end(),packed.begin()+conv_n);
            std::copy(p.dt.begin(),p.dt.end(),packed.begin()+conv_n+ab_n);
            std::copy(p.norm.begin(),p.norm.end(),packed.begin()+conv_n+2*ab_n);
            check(htod(slot.params,packed.data(),slot.params_n*sizeof(float)),"upload recurrent params");
        }
        Ptr convw_p=slot.params, a_p=slot.params+conv_n*sizeof(float);
        Ptr dt_p=a_p+ab_n*sizeof(float), norm_p=dt_p+ab_n*sizeof(float);
        buffer(rb_dn,rb_dn_size,size_t(value_dim)*sizeof(float));
        int c=channels, cw=p.conv_width, sd=p.state_dim, kh=p.key_heads, vh=p.value_heads, kd=key_dim;
        float eps=p.eps;
        { void* args[]={&qkv_p,&convw_p,&conv_state,&c,&cw};
          check(launch(delta_conv_kernel,(channels+255)/256,1,1,256,1,1,0,nullptr,args,nullptr),"launch delta conv"); }
        { void* args[]={&qkv_p,&alpha_p,&beta_p,&a_p,&dt_p,&rec_state,&rb_dn,&kd,&sd,&kh,&vh,&eps};
          check(launch(delta_scan_kernel,(vh*sd+7)/8,1,1,256,1,1,0,nullptr,args,nullptr),"launch delta scan"); }
        { void* args[]={&rb_dn,&gate_p,&norm_p,&eps,&sd};
          check(launch(delta_norm_kernel,vh,1,1,128,1,1,0,nullptr,args,nullptr),"launch delta norm"); }
        buffer(output,output_size,p.out.size_bytes());
        device_linear(*p.out_w,wo,rb_dn,output);
        check(dtoh(p.out.data(),output,p.out.size_bytes()),"read recurrent output");
    }
};
}
std::unique_ptr<Backend> cuda_backend(uint64_t budget, bool quantize) { return std::make_unique<Cuda>(budget,quantize); }
}

