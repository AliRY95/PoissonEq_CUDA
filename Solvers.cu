/* -------------------------------------------------------------------------- */
/*                            Solver Functions                                */
/*                       Alireza Yazdani, March 2021                          */
/* -------------------------------------------------------------------------- */
#ifndef SOLVERS_CU
#define SOLVERS_CU

#include <iostream>
#include <cmath>
#include <cublas_v2.h>

#include "LinearAlgebraCPU.cpp"
#include "LinearAlgebraGPU.cu"

using namespace std;
/* -------------------------------------------------------------------------- */
/*                            Global Variables                                */
extern const double TOL;
extern const int MAX_ITER;
extern int BLOCKSIZE, GRIDSIZE;
extern const double Lx, Ly;
extern int Nx, Ny;
/* -------------------------------------------------------------------------- */
// Jacobi Solver Updater on CPU
void JacobiUpdater(double* solution, double* RHS, double* dxdy,
                   int VecSize, int Stride)
{
  double temp;
  for(int i = 0; i < VecSize; i++) {
    temp = RHS[i];
    if(i >= Stride)
      temp += solution[i-Stride]/dxdy[1];
    if(i < VecSize - Stride)
      temp += solution[i+Stride]/dxdy[1];
    if(i % Stride != 0)
      temp += solution[i-1]/dxdy[0];
    if(i % Stride != Stride-1)
      temp += solution[i+1]/dxdy[0];
    temp /= (2/dxdy[0] + 2/dxdy[1]);
    solution[i] = temp;
  }
}
/* -------------------------------------------------------------------------- */
// Jacobi Solver on CPU
void JacobiSolverCPU(double* solution, double* RHS)
{
  // Defining Variables
  const int N = (Nx-2)*(Ny-2); // Length of vectors
  double* r; // Residual vector
  double* temp_vec; // Temporary vector
  double err = 1.0; // Error
  int ctr = 1; // Counter
  double dxdy[2] = {Lx*Lx/(Nx-1)/(Nx-1),Ly*Ly/(Ny-1)/(Ny-1)};

  // Allocating Dynamic Memory for Variables
  r = new double [N];
  temp_vec = new double [N];
  VecAssign(temp_vec, solution, N);

  do {
    JacobiUpdater(solution, RHS, dxdy, N, Nx-2);
    VecLinearOperator(r, solution, -1.0, temp_vec, N);
    CalcNorm(&err, r, N);
    if(ctr % 100 == 0) {
      cout << "Iter. " << ctr << ", Error " << err << '\n';
    }

    VecAssign(temp_vec, solution, N);

    ctr++;
  } while(err > TOL && ctr < MAX_ITER);

  // Freeing Dynamic Memory
  delete[] r;
  delete[] temp_vec;
}
/* -------------------------------------------------------------------------- */
// Conjugate Gradient Solver on CPU
void ConjugateGradientSolverCPU(double* solution,
                                const double* StiffMat, const double* RHS)
{
  // Defining Variables
  const int N = (Nx-2)*(Ny-2); // Length of vectors
  double* r; // Residual vector
  double* p; // Basis vector
  double* temp_vec; // Temporary vector
  double alpha = 0.0, beta = 0.0; // Conjugate gradients variables
  double err = 1.0; // Error
  double temp = 0.0; // Temporary variable
  int ctr = 1; // Counter

  // Allocating Dynamic Memory for Variables
  r = new double [N];
  p = new double [N];
  temp_vec = new double [N];

  // Initializing Vectors
  MultiplySpMatVec(temp_vec, StiffMat, solution, N, Nx-2);
  VecLinearOperator(r, RHS, -1.0, temp_vec, N);
  VecAssign(p, r, N);

  do {
    VecDot(&temp, r, r, N);
    MultiplySpMatVec(temp_vec, StiffMat, p, N, Nx-2); // temp_vec = A.p_k
    VecDot(&alpha, p, temp_vec, N);
    alpha = temp/alpha; // p_k^T.temp_vec = p_k^T.A.p_k

    VecLinearOperator(solution, solution, alpha, p, N);

    VecLinearOperator(r, r, -1.0*alpha, temp_vec, N);
    CalcNorm(&err, r, N);
    if(ctr % 100 == 0) {
      cout << "Iter. " << ctr << ", Error " << err << '\n';
    }

    VecDot(&beta, r, r, N);
    beta /= temp;

    VecLinearOperator(p, r, beta, p, N);

    ctr++;
  } while(err > TOL && ctr < MAX_ITER);

  // Freeing Dynamic Memory
  delete[] r;
  delete[] p;
  delete[] temp_vec;
}
/* -------------------------------------------------------------------------- */
// Jacobi Solver Updater
__global__ void JacobiUpdaterGPU(double* new_solution, double* solution,
                                 double* RHS, double* dxdy,
                                 int VecSize, int Stride)
{
  double temp;
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx > VecSize)
    return;
  temp = RHS[idx];
  if(idx >= Stride)
    temp += solution[idx-Stride]/dxdy[1];
  if(idx < VecSize - Stride)
    temp += solution[idx+Stride]/dxdy[1];
  if(idx % Stride != 0)
      temp += solution[idx-1]/dxdy[0];
  if(idx % Stride != Stride-1)
      temp += solution[idx+1]/dxdy[0];
  temp /= (2/dxdy[0] + 2/dxdy[1]);
  __syncthreads();
  // printf("%d\n", temp);
  new_solution[idx] = temp;
}
/* -------------------------------------------------------------------------- */
void JacobiSolverGPU(double* solution, double* RHS)
{
  const int N = (Nx-2)*(Ny-2); // Length of vectors
  const size_t MEMORY_SIZE = sizeof(double)*N; // Size of vectors

  double* r; // Residual vector
  double* temp_vec; // Temporary vector
  double* one; // one
  double h_dxdy[2] = {Lx*Lx/(Nx-1)/(Nx-1),Ly*Ly/(Ny-1)/(Ny-1)};
  double* d_dxdy; // dx^2 and dy^2
  double h_err, * d_err;
  int ctr = 1; // Counter

  // Allocating Dynamic Memory for Variables
  cudaMalloc((void**)&r, MEMORY_SIZE);
  cudaMalloc((void**)&temp_vec, MEMORY_SIZE);
  cudaMemcpy(temp_vec, solution, MEMORY_SIZE, cudaMemcpyDeviceToDevice);
  cudaMalloc((void**)&d_err, sizeof(double));
  cudaMalloc((void**)&one, sizeof(double));
  cudaMalloc((void**)&d_dxdy, sizeof(double)*2);
  cudaMemcpy(d_dxdy, h_dxdy, sizeof(double)*2, cudaMemcpyHostToDevice);
  cudaMemset(one, 1.0, sizeof(double));
  cudaMemset(d_err, 1.0, sizeof(double));

  // CuBlas Variables
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

  do {
    JacobiUpdaterGPU<<<GRIDSIZE,BLOCKSIZE>>>(temp_vec, solution,
                                             RHS, d_dxdy, N, Nx-2);
    VecLinearOperatorGPU<<<GRIDSIZE,BLOCKSIZE>>>(r, solution, -1, one, temp_vec, N);
    cublasDnrm2(handle, N, r, 1, d_err);
    cudaMemcpy(&h_err, d_err, sizeof(double), cudaMemcpyDeviceToHost);
    if(ctr % 100 == 0) {
      cout << "Iter. " << ctr << ", Error " << h_err << '\n';
    }

    VecAssignGPU<<<GRIDSIZE,BLOCKSIZE>>>(solution, temp_vec, N);
    cudaDeviceSynchronize();
    ctr++;
  } while(h_err > TOL && ctr < MAX_ITER);

  // Freeing Dynamic Memory
  cublasDestroy(handle);
  delete[] r;
  delete[] temp_vec;
}
/* -------------------------------------------------------------------------- */
// Conjugate Gradient Solver On GPU
void ConjugateGradientSolverGPU(double* solution,
                                double* StiffMat, double* RHS)
{
  const int N = (Nx-2)*(Ny-2); // Length of vectors
  const size_t MEMORY_SIZE = sizeof(double)*N; // Size of vectors

  double* r; // Residual vector
  double* p; // Basis vector
  double* temp_vec; // Temporary vector
  double* alpha, * beta, * temp, * one; // Conjugate gradients variables
  double* d_err, h_err;

  // Allocating Dynamic Memory for Variables
  cudaMalloc((void**)&r, MEMORY_SIZE);
  cudaMalloc((void**)&p, MEMORY_SIZE);
  cudaMalloc((void**)&temp_vec, MEMORY_SIZE);
  cudaMalloc((void**)&alpha, sizeof(double));
  cudaMalloc((void**)&beta, sizeof(double));
  cudaMalloc((void**)&temp, sizeof(double));
  cudaMalloc((void**)&one, sizeof(double));
  cudaMalloc((void**)&d_err, sizeof(double));
  cudaMemset(one, 1.0, sizeof(double));
  cudaMemset(d_err, 1.0, sizeof(double));

  // Initializing Vectors
  MultiplySpMatVecGPU<<<GRIDSIZE,BLOCKSIZE>>>(temp_vec, StiffMat, solution, N, Nx-2);
  VecLinearOperatorGPU<<<GRIDSIZE,BLOCKSIZE>>>(r, RHS, -1.0, one, temp_vec, N);
  VecAssignGPU<<<GRIDSIZE,BLOCKSIZE>>>(p, r, N);

  // CuBlas Variables
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

  int ctr = 1; // Counter
  do {
    cublasDdot(handle, N, r, 1, r, 1, temp);
    MultiplySpMatVecGPU<<<GRIDSIZE,BLOCKSIZE>>>(temp_vec, StiffMat, p, N, Nx-2); // temp_vec = A.p_k
    cublasDdot(handle, N, p, 1, temp_vec, 1, alpha);
    DivideScalar<<<1,1>>>(alpha, temp, alpha); // p_k^T.temp_vec = p_k^T.A.p_k

    VecLinearOperatorGPU<<<GRIDSIZE,BLOCKSIZE>>>(solution, solution, 1.0, alpha, p, N);

    VecLinearOperatorGPU<<<GRIDSIZE,BLOCKSIZE>>>(r, r, -1.0, alpha, temp_vec, N);
    cublasDnrm2(handle, N, r, 1, d_err);
    cudaMemcpy(&h_err, d_err, sizeof(double), cudaMemcpyDeviceToHost);
    if(ctr % 100 == 0) {
      cout << "Iter. " << ctr << ", Error " << h_err << '\n';
    }

    cublasDdot(handle, N, r, 1, r, 1, beta);
    DivideScalar<<<1,1>>>(beta, beta, temp);

    VecLinearOperatorGPU<<<GRIDSIZE,BLOCKSIZE>>>(p, r, 1.0, beta, p, N);
    cudaDeviceSynchronize();
    ctr++;
  } while(h_err > TOL && ctr < MAX_ITER);

  // Freeing Dynamic Memory
  cublasDestroy(handle);
  cudaFree(r);
  cudaFree(p);
  cudaFree(temp_vec);
}
#endif
