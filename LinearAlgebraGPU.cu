/* -------------------------------------------------------------------------- */
/*                        Linear Algebra Functions                            */
/*                      Alireza Yazdani, March 2021                           */
/* -------------------------------------------------------------------------- */
#ifndef LINEARALGEBRAGPU_CU
#define LINEARALGEBRAGPU_CU

#include <stdio.h>
#include <cmath>

using namespace std;
/* -------------------------------------------------------------------------- */
// Multiply Sparse Square Matrix by Vector with size VecSize: out = mat*vec
__global__ void MultiplySpMatVecGPU(double* out,
                                    double* mat, double* vec,
                                    const int VecSize, const int Stride)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx > VecSize)
    return;
  if(idx == 0)
    out[0] = mat[0 + 2]*vec[0] + mat[0 + 3]*vec[1] + mat[0 + 4]*vec[Stride];
  else if(idx > 0 && idx < Stride)
    out[idx] = mat[5*idx + 1]*vec[idx-1] + mat[5*idx + 2]*vec[idx]
             + mat[5*idx + 3]*vec[idx+1] + mat[5*idx + 4]*vec[idx+Stride];
  else if(idx >= VecSize - Stride && idx < VecSize - 1)
    out[idx] = mat[5*idx + 0]*vec[idx-Stride] + mat[5*idx + 1]*vec[idx-1]
             + mat[5*idx + 2]*vec[idx] + mat[5*idx + 3]*vec[idx+1];
  else if(idx == VecSize - 1)
    out[VecSize-1] = mat[5*(VecSize-1) + 0]*vec[VecSize-1-Stride]
                   + mat[5*(VecSize-1) + 1]*vec[VecSize-1-1]
                   + mat[5*(VecSize-1) + 2]*vec[VecSize-1];
  else
    out[idx] = mat[5*idx + 0]*vec[idx-Stride]
             + mat[5*idx + 1]*vec[idx-1]
             + mat[5*idx + 2]*vec[idx]
             + mat[5*idx + 3]*vec[idx+1]
             + mat[5*idx + 4]*vec[idx+Stride];
}
/* -------------------------------------------------------------------------- */
// Two Vectors (Size VecSize) Linear Operation: out = u + c*v
__global__ void VecLinearOperatorGPU(double* out,
                                     double* u,
                                     const double c, const double* var,
                                     double* v,
                                     const int VecSize)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < VecSize)
    out[idx] = u[idx] + c*var[0]*v[idx];
}
/* -------------------------------------------------------------------------- */
// Vector to Vector (Size N) Assign Operation: v = u
__global__ void VecAssignGPU(double* out, double* u, const int VecSize)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < VecSize)
    out[idx] = u[idx];
}
/* -------------------------------------------------------------------------- */
// Divide Two Scalars on GPU
__global__ void DivideScalar(double* out, double* u, double* v)
{
  out[0] = u[0]/v[0];
}



//Print
__global__ void MyPrintGPU(double* u, int VecSize) {
  for (int i = 0; i < VecSize; i++) {
    printf("%f\n", u[i]);
  }
}
/* -------------------------------------------------------------------------- */
// Naive implementation of reduction algorithm
// __device__ int SumGPU(volatile int* shArr) {
//     int idx = threadIdx.x % 32; //the lane index in the warp
//     if (idx<16) {
//       shArr[idx] += shArr[idx+16];
//       shArr[idx] += shArr[idx+8];
//       shArr[idx] += shArr[idx+4];
//       shArr[idx] += shArr[idx+2];
//       shArr[idx] += shArr[idx+1];
//     }
//     return shArr[0];
// }
//
// __global__ void VecDotGPU(double *out,
//                           double *u, double *v, const int VecSize)
// {
//     int idx = threadIdx.x;
//     int sum = 0;
//     for (int i = idx; i < VecSize; i += BLOCKSIZE)
//         sum += u[i]*v[i];
//     __shared__ int r[BLOCKSIZE];
//     r[idx] = sum;
//     SumGPU(&r[idx & ~31]);
//     __syncthreads();
//     if (idx<warpSize) { //first warp only
//         r[idx] = 32*idx < BLOCKSIZE ? r[32*idx] : 0;
//         SumGPU(r);
//         if (idx == 0)
//             *out = r[0];
//     }
// }
#endif
