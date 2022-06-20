/* -------------------------------------------------------------------------- */
/* Solving Poisson Equation (\Delta u = f) w/ Dirichlet BC Using NVIDIA CUDA. */
/*                     Alireza Yazdani, March 2021                            */
/* -------------------------------------------------------------------------- */
#include <stdio.h>
#include <new>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include "cublas_v2.h"

using namespace std;

/* Global Variables */
#define Lx (2*M_PI) // Length
#define Ly (2*M_PI) // Width

__managed__ int Nx; // Number of Pts in x-direction
__managed__ int Ny; // Number of Pts in y-direction

__managed__ double dx; // Grid Size, x-direction
__managed__ double dy; // Grid Size, y-direction

const double REL_TOL = 1e-8; // Tolerance
const int MAX_ITER = 354872; // Tolerance
/* -------------------------------------------------------------------------- */
                              /* Functions */
// RHS of The Poisson Equation
double f_function(double x, double y) {return 2*sin(x)*cos(y);}
// Dirichlet Boundary Conditions
double g_function(double x, double y) {return sin(x)*cos(y);}
// Exact Solution for Comparison
double exact_function(double x, double y) {return sin(x)*cos(y);}
// Writing Output to .dat Files
void WriteOutput(double* x, double* y, double* u, int iter, double error, double time)
{
  ofstream conv_data;
  conv_data.open("Data/conv_data_GPU_v3.dat", ios::app);
  conv_data << Nx << '\t' << Ny << '\t'
            << iter << '\t' << error << '\t' << time << '\n';
  conv_data.close();
  cout << "\n Mesh:" << Nx << "*" << Ny
       << ", \t No. Iter.: " << iter
       << ", \t Error: " << error
       << ", \t Time: " << time << ".\n";
  ofstream solution;
  solution.open("Data/sol_GPU.dat", ios::trunc);
  for(int j = 0; j < Ny; j++) {
    for(int i = 0; i < Nx; i++) {
      solution << x[i] << '\t' << y[j] << '\t' << u[i + Nx*j] << '\n';
    }
  }
}
// Calculating Max Norm
double CalculateNorm(double* vec) {
  double norm = 0.0;
  for(int i = 0; i < Nx*Ny ; i++) {
    norm = (vec[i] > norm ? vec[i] : norm);
  }
  return norm;
}

                               /* Kernels */
// Updating Solution Using Point Jacobi
__global__ void UpdateSolution(double* solution, double* RHS)
{
  double temp;
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx > Nx && idx < (Ny-1)*Nx && idx % Nx != 0 && idx % Nx != Nx-1) {
    // Update Solution
    temp = (RHS[idx] +
          (solution[idx+1] + solution[idx-1])/(dx*dx) +
          (solution[idx+Ny] + solution[idx-Ny])/(dy*dy))/
          (2/dx/dx + 2/dy/dy);
    __syncthreads();
    solution[idx] = temp;
  }
}
// Subtracting Two Vectors
__global__ void AddVectors(double* diff, const double* vec1, const double c, const double* vec2) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  diff[idx] = vec1[idx] + c*vec2[idx];
}
/* -------------------------------------------------------------------------- */
/* Main Function */
int main(int argc, char* argv[])
{
  // Initializing Global Variables
  Nx = stoi(argv[1]);
  Ny = Nx;
  dx = Lx/(Nx-1);
  dy = Ly/(Ny-1);

  // Defining Variables
  double* x; // x-coordinates
  double* y; // y-coordinates
  double* h_Solution; // Solution on host
  double* h_ExactSolution; // Exact solution on host
  double* h_RHS; // RHS of equation (f function) on host
  double* d_Solution; // Solution on device
  double* d_ExactSolution; // Exact solution on device
  double* d_RHS; // RHS of equation (f function) on device
  double* d_Error; // Change in solution in one step on device
  long MEMORY_SIZE = ((int)sizeof(double))*Nx*Ny; // Size of solution vector
  const int NUM_THREADS = 32*4; // No. Threads
  const int NUM_BLOCKS = Nx*Ny/NUM_THREADS; // No. Blocks

  // Allocating Dynamic Memory for variables
  x = new double [Nx];
  y = new double [Ny];
  h_Solution = new double [Nx*Ny];
  h_ExactSolution = new double [Nx*Ny];
  h_RHS = new double [Nx*Ny];
  cudaMalloc((void**)&d_Solution, MEMORY_SIZE);
  cudaMalloc((void**)&d_ExactSolution, MEMORY_SIZE);
  cudaMalloc((void**)&d_RHS, MEMORY_SIZE);
  cudaMalloc((void**)&d_Error, MEMORY_SIZE);

  // Coordinates
  for(int i = 0; i < Nx; i++)
    x[i] = i*dx;
  for(int j = 0; j < Ny; j++)
    y[j] = j*dy;
  // Initializing Solution, Exact Solution, RHS Function
  int idx;
  for(int j = 0; j < Ny; j++) {
    for(int i = 0; i < Nx; i++) {
      idx = i + Nx*j;
      h_Solution[idx] = 0.0;
      h_ExactSolution[idx] = exact_function(x[i], y[j]);
      h_RHS[idx] = f_function(x[i], y[j]);
    }
  }
  // Top and Bottom BCs
  for(int i = 0; i < Nx; i++) {
    h_Solution[i] = g_function(x[i], y[0]);
    h_Solution[i + Nx*(Ny-1)] = g_function(x[i], y[Ny-1]);
  }
  // Left and Right BCs
  for(int j = 0; j < Ny; j++) {
    h_Solution[Nx*j] = g_function(x[0], y[j]);
    h_Solution[Nx - 1 + Nx*j] = g_function(x[Nx-1], y[j]);
  }

  // Copying solution from host to device, Initializing device variables
  cudaMemcpy(d_Solution, h_Solution, MEMORY_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ExactSolution, h_ExactSolution, MEMORY_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_RHS, h_RHS, MEMORY_SIZE, cudaMemcpyHostToDevice);
  cudaMemset(d_Error, 0, MEMORY_SIZE);

  // Iterations
  auto h_start = std::chrono::high_resolution_clock::now();
  double err1 = 1.0;
  double err2 = 0.;
  int ctr = 1;
  do {
    // err2 = err1;
    UpdateSolution <<<NUM_BLOCKS, NUM_THREADS>>> (d_Solution, d_RHS);
    AddVectors <<<NUM_BLOCKS, NUM_THREADS>>>(d_Error, d_ExactSolution, -1, d_Solution);
    cudaMemcpy(h_Solution, d_Error, MEMORY_SIZE, cudaMemcpyDeviceToHost);
    // err1 = CalculateNorm(h_Solution);
    if(ctr % 10000 == 0) {
      cout << "Iter. " << ctr << ", Error " << err1 << '\n';
    }
    ctr++;
  } while(abs(err1-err2) > REL_TOL && ctr < MAX_ITER);
  cudaDeviceSynchronize();
  auto h_end = std::chrono::high_resolution_clock::now();

  // Outputs
  std::chrono::duration<float> time = h_end - h_start;
  cudaMemcpy(h_Solution, d_Solution, MEMORY_SIZE, cudaMemcpyDeviceToHost);
  WriteOutput(x, y, h_Solution, --ctr, err1, time.count());

  delete[] x;
  delete[] y;
  delete[] h_Solution;
  delete[] h_ExactSolution;
  delete[] h_RHS;
  cudaFree(d_Solution);
  cudaFree(d_ExactSolution);
  cudaFree(d_RHS);
  cudaFree(d_Error);

  return 0;
}
