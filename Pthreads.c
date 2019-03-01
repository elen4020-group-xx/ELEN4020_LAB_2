#include <stdio.h>
#include <stdlib.h>
#include "rank2Tensor.h"
#include <time.h>
#include <omp.h>
#include <pthread.h>

void swap(int* i1,int* i2)
{
	 int temp =*i2;
	 *i2=*i1;
	 *i1=temp;
	
}

typedef struct forDiagonal
{
	rank2Tensor* srcMat;
	int start;
	int end;

} forDiag;



void naiveTranspose(rank2Tensor* t)
{
 	for(int i=0; i<t->rows; i++)
 	{	
 		for(int j=0; j<i; j++)
 		{
 			swap(&(t->matrix[i][j]),&(t->matrix[j][i]));
 		}

 	}

}

void* DiagTranspose(void* arg)
{

	forDiag* fd= (forDiag*) arg;
	rank2Tensor* t= fd->srcMat;

	for(int i=fd->start;i<fd->end;i++)
	{
		for(int j=i;j<t->cols;j++)
		{
			swap(&(t->matrix[i][j]),&(t->matrix[j][i]));	
		}
	}

}

void blockTranspose(rank2Tensor* t)
{

	for(int i=0; i<t->rows; i+=2)
	{


		for(int j=i; j<t->cols; j+=2)
		{
			//internal transpose
			swap(&(t->matrix[i][j+1]),&(t->matrix[i+1][j]));
			//swap(&(t->matrix[j][i+1]),&(t->matrix[j+1][i]));



			if(i!=j)
			{
				swap(&(t->matrix[j][i+1]),&(t->matrix[j+1][i]));
				swap(&(t->matrix[i][j]),&(t->matrix[j][i]));
				swap(&(t->matrix[i][j+1]),&(t->matrix[j][i+1]));		
				swap(&(t->matrix[i+1][j]),&(t->matrix[j+1][i]));
				swap(&(t->matrix[i+1][j+1]),&(t->matrix[j+1][i+1]));	
			}

			///
			
		}



	}
}

int main ()
{
	srand(time(NULL));

	
	const int numThreads=4;
	rank2Tensor t;


	int N_0=1024;
	t.rows=N_0;
	t.cols=N_0;
	initRank2Tensor(&t);


	forDiag argsForDiag[numThreads];

	pthread_t threads[numThreads];


	argsForDiag[0].srcMat=&t;
	argsForDiag[0].start=0;
	argsForDiag[0].end=N_0/numThreads;

	for (int i = 1; i < numThreads-1; ++i)
	{
		argsForDiag[i].srcMat=&t;
		argsForDiag[i].start=argsForDiag[i-1].end;
		argsForDiag[i].end=argsForDiag[i].start+N_0/numThreads;

	}
	argsForDiag[numThreads-1].srcMat=&t;
	argsForDiag[numThreads-1].start=argsForDiag[numThreads-2].end;
	argsForDiag[numThreads-1].end=N_0;

	printf("%d\n", argsForDiag[numThreads-1].end);


	for (int i = 0; i<numThreads; i++)
	{
		pthread_create(&threads[i],NULL,&DiagTranspose,&argsForDiag[i]);
	}
	for (int i = 0; i<numThreads; i++)
	{
		pthread_join(threads[i],NULL);
	}

	double time = omp_get_wtime();
	DiagTranspose(&t);

	double time2 = omp_get_wtime() - time;
	printf("time elapsed (Diag) : %f\n",((float)time2)/1);


	disposeRank2Tensor(&t);

	return 0;
}