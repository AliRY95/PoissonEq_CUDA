To compile:
	nvcc Poisson.cu -o Poisson -lcublas
	
To run:
	./Poisson CPU(/GPU) 64(No. points)  CG(/J)
	
Example:
1. Run conjugate gradient for a problem with 128 points in each direction on GPU:
	./Poisson GPU 128 CG
2. Run Jacobi for a problem with 64 points in each direction on CPU:
	./Poisson CPU 64 J
