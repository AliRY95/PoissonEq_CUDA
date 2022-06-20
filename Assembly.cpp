/* -------------------------------------------------------------------------- */
/*                   Poisson Problem Related Functions                        */
/*                      Alireza Yazdani, March 2021                           */
/* -------------------------------------------------------------------------- */
#ifndef ASSEMBLY_CPP
#define ASSEMBLY_CPP

#include <cmath>
#include <iostream>
#include <fstream>

#include "LinearAlgebraCPU.cpp"

using namespace std;
/* -------------------------------------------------------------------------- */
/*                            Global Variables                                */
extern const double Lx, Ly;
extern int Nx, Ny;
/* -------------------------------------------------------------------------- */
// RHS of The Poisson Equation
double f_function(const double x, const double y) {return 2*sin(x)*cos(y);}
// Dirichlet Boundary Conditions
double g_function(const double x, const double y) {return sin(x)*cos(y);}
// Exact Solution for Comparison
double exact_function(const double x, const double y) {return sin(x)*cos(y);}
/* -------------------------------------------------------------------------- */
// Poisson Problem Variables
void AssembleProblem5Pts(double* StiffMat, double* Solution, double* RHS,
                         double* x, double* y)
{
  const double dx = Lx/(Nx-1);
  const double dy = Ly/(Ny-1);
  // Coordinates
  for(int i = 0; i < Nx-2; i++)
    x[i] = (i+1)*dx;
  for(int j = 0; j < Ny-2; j++)
    y[j] = (j+1)*dy;
  // Initializing Solution, and Computing StiffMat, and RHS
  int i, j, idx;
  for(j = 0; j < Ny-2; j++) {
    for(i = 0; i < Nx-2; i++) {
      idx = i + (Nx-2)*j;
      Solution[idx] = 0.0;
      StiffMat[5*idx+0] = -1/dy/dy;
      StiffMat[5*idx+1] = -1/dx/dx;
      StiffMat[5*idx+2] = 2/dx/dx + 2/dy/dy;
      StiffMat[5*idx+3] = -1/dx/dx;
      StiffMat[5*idx+4] = -1/dy/dy;
      RHS[idx] = f_function(x[i], y[j]);
    }
  }
  // Boundary Conditions
  // Left and Right BC
  for(j = 0; j < Ny-2; j++) {
    StiffMat[5*((Nx-2)*j) + 1] = 0.0;
    RHS[(Nx-2)*j] += g_function(0, y[j])/dx/dx;
    StiffMat[5*((Nx-3) + (Nx-2)*j) + 3] = 0.0;
    RHS[(Nx-3) + (Nx-2)*j] += g_function(Lx, y[j])/dx/dx;
  }
  // Top and Bottom BC
  for(i = 0; i < Nx-2; i++) {
    StiffMat[5*i + 0] = 0.0;
    RHS[i] += g_function(x[i], 0)/dy/dy;
    StiffMat[5*(i + (Nx-2)*(Ny-3)) + 4] = 0.0;
    RHS[i + (Nx-2)*(Ny-3)] += g_function(x[i], Ly)/dy/dy;
  }
}
/* -------------------------------------------------------------------------- */
// Writing Output to .dat Files
void WriteOutput(string Mode, string Solver,
                 double time, double* x, double* y, double* u)
{
  // Calculating Exact Error
  double* Error; // Exact solution on host
  Error = new double [(Nx-2)*(Ny-2)]; // Dynamic memory
  double err;
  for(int j = 0; j < Ny-2; j++) {
    for(int i = 0; i < Nx-2; i++) {
      Error[i + (Nx-2)*j] = exact_function(x[i], y[j]) - u[i + (Nx-2)*j];
    }
  }
  CalcNorm(&err, Error, (Nx-2)*(Ny-2));

  ofstream conv_data;
  string ConvFile = "Data/" + Mode + Solver + "Conv.dat";
  conv_data.open(ConvFile, ios::app);
  conv_data << Nx << '\t' << Ny << '\t' << err << '\t' << time << '\n';
  conv_data.close();
  cout << "\nMesh:" << Nx << "*" << Ny
       << ", \t Error: " << err
       << ", \t Time: " << time << ".\n";

  ofstream solution;
  string SolFile = "Data/" + Mode + to_string(Nx) + Solver + "Sol.dat";
  solution.open(SolFile, ios::trunc);
  for(int j = 0; j < Ny-2; j++) {
    for(int i = 0; i < Nx-2; i++) {
      solution << x[i] << '\t' << y[j] << '\t' << u[i + (Nx-2)*j] << '\n';
    }
  }
  solution.close();

  delete[] Error;
}
#endif
