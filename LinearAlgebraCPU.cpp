/* -------------------------------------------------------------------------- */
/*                        Linear Algebra Functions                            */
/*                      Alireza Yazdani, March 2021                           */
/* -------------------------------------------------------------------------- */
#ifndef LINEARALGEBRACPU_H
#define LINEARALGEBRACPU_H

#include <cmath>
/* -------------------------------------------------------------------------- */
// Multiply Sparse Square Matrix by Vector with size VecSize: out = mat*vec
void MultiplySpMatVec(double* out,
                      const double* mat, const double* vec,
                      const int VecSize, const int Stride)
{
  out[0] = mat[5*0+2] * vec[0]
         + mat[5*0+3] * vec[1]
         + mat[5*0+4] * vec[Stride];
  for(int i = 1; i < Stride; i++) {
    out[i] = mat[5*i+1] * vec[i-1]
           + mat[5*i+2] * vec[i]
           + mat[5*i+3] * vec[i+1]
           + mat[5*i+4] * vec[i+Stride];
  }
  for(int i = Stride; i < VecSize - Stride; i++) {
    out[i] = mat[5*i+0] * vec[i-Stride]
           + mat[5*i+1] * vec[i-1]
           + mat[5*i+2] * vec[i]
           + mat[5*i+3] * vec[i+1]
           + mat[5*i+4] * vec[i+Stride];
  }
  for(int i = VecSize - Stride; i < VecSize-1; i++) {
    out[i] = mat[5*i+0] * vec[i-Stride]
           + mat[5*i+1] * vec[i-1]
           + mat[5*i+2] * vec[i]
           + mat[5*i+3] * vec[i+1];
  }
  out[VecSize-1] = mat[5*(VecSize-1)+0] * vec[VecSize-1-Stride]
                 + mat[5*(VecSize-1)+1] * vec[VecSize-1-1]
                 + mat[5*(VecSize-1)+2] * vec[VecSize-1];
}
/* -------------------------------------------------------------------------- */
// Dot Product of Two Vectors with Size VecSize
void VecDot(double* out,
            const double* u, const double* v, const int VecSize)
{
  double DotProduct = 0.0;
  for(int i = 0; i < VecSize; i++)
    DotProduct += u[i]*v[i];
  *out = DotProduct;
}
/* -------------------------------------------------------------------------- */
// Two Vectors (Size N) Linear Operation: out = u + c*v
void VecLinearOperator(double* out,
                       const double* u, const double c, const double* v,
                       const int VecSize)
{
  for(int i = 0; i < VecSize; i++)
    out[i] = u[i] + c*v[i];
}
/* -------------------------------------------------------------------------- */
// Vector to Vector (Size N) Assign Operation
void VecAssign(double* out, const double* u, const int VecSize)
{
  for(int i = 0; i < VecSize; i++)
    out[i] = u[i];
}
/* -------------------------------------------------------------------------- */
// L2-Norm of Vector with Size VecSize
void CalcNorm(double* out, double* u, const int VecSize)
{
  double norm = 0.0;
  for(int i = 0; i < VecSize; i++)
    // norm = (abs(u[i]) > norm ? abs(u[i]) : norm);
    norm += u[i]*u[i];
  *out = sqrt(norm/VecSize);
}

void MyPrint(double* u, int VecSize) {
  for (int i = 0; i < VecSize; i++) {
    printf("%f\n", u[i]);
  }
}
#endif
