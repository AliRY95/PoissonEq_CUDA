/* -------------------------------------------------------------------------- */
/* Solving Poisson Equation (-\Delta u = f) w/ Dirichlet BC Using NVIDIA CUDA.*/
/*                      Alireza Yazdani, March 2021                           */
/* -------------------------------------------------------------------------- */
#include <stdio.h>
#include <cmath>
#include <new>
#include <chrono>
#include <string>

#include "Assembly.cpp"
#include "LinearAlgebraCPU.cpp"
#include "LinearAlgebraGPU.cu"
#include "Solvers.cu"

using namespace std;
/* -------------------------------------------------------------------------- */
/*                            Global Variables                                */
// Dimensions
const double Lx = (2*M_PI); // Rectangle width
const double Ly = (2*M_PI); // Rectangle length

// Grid
int Nx; // Number of grid points in x-direction
int Ny; // Number of grid points in y-direction

// Solver Parameters
const double TOL = 1e-8; // Tolerance
const int MAX_ITER = 1e6; // Maximum No. iterations
int BLOCKSIZE; // Threads in a block
int GRIDSIZE; // No. blocks
/* -------------------------------------------------------------------------- */
/* Main Function */
int main(int argc, char* argv[])
{
  // Command Line Inputs
  string temp = "GPU";
  bool Mode = (argv[1] == temp);
  Nx = Ny = stoi(argv[2]);
  temp = "CG";
  bool Solver = (argv[3] == temp);

  // Defining Variables
  int N = (Nx-2)*(Ny-2); // length of vectors excluding boundary pts
  BLOCKSIZE = 64;
  GRIDSIZE = ceil((double)N/BLOCKSIZE);

  // Defining Arrays
  double* x; // x-coordinates
  double* y; // y-coordinates
  double* h_Solution; // Solution on host
  double* h_StiffMat; // Stiffness matrix on host
  double* h_RHS; // RHS of equation on host
  double* d_Solution; // Solution on device
  double* d_StiffMat; // Stiffness matrix on device
  double* d_RHS; // RHS of equation on device
  const size_t MEMORY_SIZE = sizeof(double)*N; // Size of vectors in bytes

  // Allocating Dynamic Memory for Host Variables
  x = new double [Nx-2];
  y = new double [Ny-2];
  h_Solution = new double [N];
  h_StiffMat = new double [5*N];
  h_RHS = new double [N];

  // Assembeling Matrices on Host
  AssembleProblem5Pts(h_StiffMat, h_Solution, h_RHS, x, y);

  if(Mode == 1)
  {
    // Allocating Dynamic Memory for Device Variables
    cudaMalloc((void**)&d_Solution, MEMORY_SIZE);
    cudaMalloc((void**)&d_StiffMat, 5*MEMORY_SIZE);
    cudaMalloc((void**)&d_RHS, MEMORY_SIZE);

    // Copying Host Arrays to Device Arrays
    cudaMemcpy(d_StiffMat, h_StiffMat, 5*MEMORY_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Solution, h_Solution, MEMORY_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_RHS, h_RHS, MEMORY_SIZE, cudaMemcpyHostToDevice);
  }

  // Iterations
  auto h_start = chrono::high_resolution_clock::now();
  if(Mode == 0 && Solver == 0)
    JacobiSolverCPU(h_Solution, h_RHS);
  else if(Mode == 0 && Solver == 1)
    ConjugateGradientSolverCPU(h_Solution, h_StiffMat, h_RHS);
  else if(Mode == 1 && Solver == 0)
    JacobiSolverGPU(d_Solution, d_RHS);
  else if(Mode == 1 && Solver == 1)
    ConjugateGradientSolverGPU(d_Solution, d_StiffMat, d_RHS);
  auto h_end = chrono::high_resolution_clock::now();

  // Outputs
  chrono::duration<float> time = h_end - h_start; // Copmuting Time
  if(Mode == 1)
  {
    cudaMemcpy(h_Solution, d_Solution, MEMORY_SIZE, cudaMemcpyDeviceToHost);
    cudaFree(d_Solution);
    cudaFree(d_StiffMat);
    cudaFree(d_RHS);
  }
  WriteOutput(argv[1], argv[3], time.count(), x, y, h_Solution);

  delete[] x;
  delete[] y;
  delete[] h_Solution;
  delete[] h_StiffMat;
  delete[] h_RHS;

  return 0;
}
