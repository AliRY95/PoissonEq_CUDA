/* -------------------------------------------------------------------------- */
/* Solving Poisson Equation (\Delta u = f) w/ Dirichlet BC Using NVIDIA CUDA. */
/*                      Alireza Yazdani, March 2021                           */
/* -------------------------------------------------------------------------- */
#include <stdio.h>
#include <new>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>

using namespace std;

/* Global Variables */
#define Lx (2*M_PI) // Length
#define Ly (2*M_PI) // Width

int Nx; // Number of Pts in x-direction
int Ny; // Number of Pts in y-direction

double dx; // Grid Size, x-direction
double dy; // Grid Size, y-direction

const double REL_TOL = 1e-12; // Tolerance
const int MAX_ITER = 1e6; // Tolerance
/* -------------------------------------------------------------------------- */
                              /* Functions */
// RHS of The Poisson Equation
double f_function(double x, double y) {return 2*sin(x)*cos(y);}
// Dirichlet Boundary Conditions
double g_function(double x, double y) {return sin(x)*cos(y);}
// Exact Solution for Comparison
double exact_function(double x, double y) {return sin(x)*cos(y);}
// Updating Solution Using Point Jacobi
void UpdateSolution(double* solution, double* RHS)
{
  double new_solution[Nx*Ny] = {0.0};
  for(int j = 1; j < Ny-1; j++) {
    for(int i = 1; i < Nx-1; i++) {
      int idx = i + Nx*j;
      // New Solution
      new_solution[idx] = (RHS[idx] +
                          (solution[idx+1] + solution[idx-1])/(dx*dx) +
                          (solution[idx+Ny] + solution[idx-Ny])/(dy*dy))/
                          (2/dx/dx + 2/dy/dy);
      /*
      solution[idx] = (RHS[idx] +
            (solution[idx+1] + solution[idx-1])/(dx*dx) +
            (solution[idx+Ny] + solution[idx-Ny])/(dy*dy))/(2/dx/dx + 2/dy/dy);
      */
    }
  }
  // Update Solution
  for(int j = 1; j < Ny-1; j++) {
    for(int i = 1; i < Nx-1; i++) {
      int idx = i + Nx*j;
      solution[idx] = new_solution[idx];
    }
  }
}
// Linear Operation with Two Vectors
void AddVectors(double* diff, double* vec1, double c, double* vec2) {
  for(int i = 0; i < Nx*Ny ; i++) {
    diff[i] = vec1[i] + c*vec2[i];
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
// Writing Output to .dat Files
void WriteOutput(double* x, double* y, double* u, int iter, double error, double time)
{
  ofstream conv_data;
  conv_data.open("Data/conv_data_CPU.dat", ios::app);
  conv_data << Nx << '\t' << Ny << '\t'
            << iter << '\t' << error << '\t' << time << '\n';
  conv_data.close();
  cout << "\n Mesh:" << Nx << "*" << Ny
       << ", \t No. Iter.: " << iter
       << ", \t Error: " << error
       << ", \t Time: " << time << ".\n";
  ofstream solution;
  solution.open("Data/sol_CPU.dat", ios::trunc);
  for(int j = 0; j < Ny; j++) {
    for(int i = 0; i < Nx; i++) {

      solution << x[i] << '\t' << y[j] << '\t' << u[i + Nx*j] << '\n';
    }
  }
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
  double* h_Error; // Change in solution in one step

  // Allocating Dynamic Memory for Variables
  x = new double [Nx];
  y = new double [Ny];
  h_Solution = new double [Nx*Ny];
  h_ExactSolution = new double [Nx*Ny];
  h_RHS = new double [Nx*Ny];
  h_Error = new double [Nx*Ny];

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

  // Iterations
  auto h_start = std::chrono::high_resolution_clock::now();
  double err1 = 1.0;
  double err2;
  int ctr = 1;
  do {
    err2 = err1;
    UpdateSolution(h_Solution, h_RHS);
    AddVectors(h_Error, h_ExactSolution, -1, h_Solution);
    err1 = CalculateNorm(h_Error);
    if(ctr % 1 == 0) {
      cout << "Iter. " << ctr << ", Error " << err1 << '\n';
    }
    ctr++;
  } while(abs(err1-err2) > REL_TOL && ctr < MAX_ITER);
  auto h_end = std::chrono::high_resolution_clock::now();

  // Outputs
  std::chrono::duration<float> time = h_end - h_start;
  WriteOutput(x, y, h_Solution, --ctr, err1, time.count());

  delete[] x;
  delete[] y;
  delete[] h_Solution;
  delete[] h_ExactSolution;
  delete[] h_RHS;
  delete[] h_Error;

  return 0;
}
