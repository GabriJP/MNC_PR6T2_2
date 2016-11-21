#include <cstdio>
#include <random>

#include <mkl.h>
#include <omp.h>

#define Nbig 4000
#define MAX_THREADS 20
#define TST

extern void BalanceoCarga(int Nth, int M, int *Pos, int *Num);

int main(int argc, char *argv[]){
	printf("\nUso conjunto de OpenMP y BLAS en la multiplicacion de grandes matrices\nAutor: Gabriel Jimenez para MNC\n");

	int Pos[MAX_THREADS], Num[MAX_THREADS];
	int Nth, N;
	double *A, *B, *C;

#ifdef TEST
	printf("\nCaso de prueba\n\n");
	N = 5;
	A = (double *)mkl_malloc(N*N*sizeof(double), 64);
	B = (double *)mkl_malloc(N*N*sizeof(double), 64);
	C = (double *)mkl_malloc(N*N*sizeof(double), 64);

	for (int i = 0; i < N*N; i++){
		A[i] = 1.0 + (double)i;
		B[i] = (double)(N*N) - (double)i;
	}

	Nth = 3;
	BalanceoCarga(Nth, N, Pos, Num);
#else
	N = Nbig;
	A = (double *)mkl_malloc(N*N*sizeof(double), 64);
	B = (double *)mkl_malloc(N*N*sizeof(double), 64);
	C = (double *)mkl_malloc(N*N*sizeof(double), 64);

	std::default_random_engine generador;
	std::normal_distribution<double> distribucion(0.0, 1.0);
	for (int i = 0; i < N*N; i++){
		A[i] = distribucion(generador);
		B[i] = distribucion(generador);
	}

	Nth = 2;
	BalanceoCarga(Nth, N, Pos, Num);
#endif

	//fork
	int i;
	double inicio = omp_get_wtime();
#pragma omp parallel for private(i) num_threads(Nth)
	for (i = 0; i < Nth; i++){
		printf("Hilo: %d, desde %d, Nlineas: %d\n", i, Pos[i], Num[i]);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Num[i], N, N, 1.0, &(A[Pos[i]*N]), N, B, N, 0.0, &(C[Pos[i]*N]), N);
	}
	double fin = omp_get_wtime();

	//Report de resultados
#ifdef TEST
	for (int a = 0; a < N; a++){
		for (int b = 0; b < N; b++){
			printf("%g ", C[a*N+b]);
		}
		printf("\n");
	}
#else
	double tiempo = fin - inicio;
	double Gflops = 2.0*N*N*N*1.0e-09 / tiempo;
	printf("\nThread: %d, Tiempo: %g segundos, GFlops: %g\n", Nth, tiempo, Gflops);
#endif

	mkl_free(A);
	mkl_free(B);
	mkl_free(C);
	std::getchar();
	return 0;
}