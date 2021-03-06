#include <stdio.h>
#include <stdlib.h>
#include "rank2Tensor.h"
#include <time.h>
#include <omp.h>
#include <pthread.h>
#include <math.h>

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

typedef struct forBlock
{
	rank2Tensor* srcMat;
	int startBlock;
	int noBlocks;

} forBlock;




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

void* blockTranspose(void* arg)
{
	forBlock* blocks = (forBlock*)arg;
	//resolve arguments
	int blockCount = blocks->noBlocks;
	int startBlock = blocks->startBlock;
	int endBlock=blockCount+startBlock-1;


	////Convert block coordinate space to matrix coordinate space
	rank2Tensor* t = blocks->srcMat;
	int startRow = (int) ceil( (-1 +sqrt(1+(startBlock+1)*8))/2) -1;

	int startRowStart=((startRow+1)*(startRow+2))/2 - (startRow+1); //blocks per row= row index +1
	int startCol = (startBlock-startRowStart)*2;

	int endRow = (int) ceil((-1 +sqrt(1+(1+endBlock)*8))/2) -1;

	int endRowStart = ((endRow+1)*(endRow+2))/2 - (endRow+1);
	int endCol = (endBlock-endRowStart)*2;
	////

	for(int i=(startRow)*2; i<=(endRow)*2; i+=2)
	{
		int p=i;
		//if in last row only go up to some point
		if (i==((endRow)*2))
		{
			p=endCol;
		}
		for(int j=startCol; j<=p; j+=2)
		{
			//internal transpose
			swap(&(t->matrix[i][j+1]),&(t->matrix[i+1][j]));
			if(i!=j)
			{			
				swap(&(t->matrix[j][i+1]),&(t->matrix[j+1][i]));
				swap(&(t->matrix[i][j]),&(t->matrix[j][i]));
				swap(&(t->matrix[i][j+1]),&(t->matrix[j][i+1]));		
				swap(&(t->matrix[i+1][j]),&(t->matrix[j+1][i]));
				swap(&(t->matrix[i+1][j+1]),&(t->matrix[j+1][i+1]));
			}
		}
		startCol=0;
	}
}

int main ()
{
	srand(time(NULL));
	rank2Tensor t;
	char* fileName="pthreads.csv";
	FILE *fp;
	fp=fopen(fileName,"a");
	double times_diag[6]={0};
	double times_block[6]={0};


	const int matSizes[6]={128,1024,2048,4096,8196,16392};
	//foreach matrix size
	for (int testNo = 0; testNo < 6; testNo++)
	{
		int numThreads = 8;

		int N_0 = matSizes[testNo];
		t.rows = N_0;
		t.cols = N_0;
		initRank2Tensor(&t);
		//diagonal worksharing
		forDiag argsForDiag[numThreads];

		pthread_t threads[numThreads];
		argsForDiag[0].srcMat = &t;
		argsForDiag[0].start = 0;
		argsForDiag[0].end = N_0 / numThreads;

		for (int i = 1; i < numThreads - 1; ++i)
		{
			argsForDiag[i].srcMat = &t;
			argsForDiag[i].start = argsForDiag[i - 1].end;
			argsForDiag[i].end = argsForDiag[i].start + N_0 / numThreads;
		}
		argsForDiag[numThreads - 1].srcMat = &t;
		argsForDiag[numThreads - 1].start = argsForDiag[numThreads - 2].end;
		argsForDiag[numThreads - 1].end = N_0;

		printf("%d\n", argsForDiag[numThreads - 1].end);

		double time = omp_get_wtime();
		for (int i = 0; i < numThreads; i++)
		{
			pthread_create(&threads[i], NULL, &DiagTranspose, &argsForDiag[i]);
		}
		for (int i = 0; i < numThreads; i++)
		{
			pthread_join(threads[i], NULL);
		}
		double time2 = omp_get_wtime() - time;
		times_diag[testNo]=time2;
		printf("time elapsed (Diag) : %f\n", ((float)time2) / 1);

		printf("\n");
		//// Block worksharing
		int blockCount = t.rows / 2;
		blockCount = blockCount * (blockCount + 1) / 2;

		forBlock argsForBlock[numThreads];
		pthread_t threads_1[numThreads];
		if (blockCount < numThreads)
		{
			numThreads = blockCount;
		}

		int blocksPerThread = blockCount / numThreads;
		argsForBlock[0].srcMat = &t;
		argsForBlock[0].noBlocks = blocksPerThread;
		argsForBlock[0].startBlock = 0;

		for (int i = 1; i < numThreads - 1; ++i)
		{
			argsForBlock[i].srcMat = &t;
			argsForBlock[i].noBlocks = blocksPerThread;
			argsForBlock[i].startBlock = i * blocksPerThread;
		}
		argsForBlock[numThreads - 1].srcMat = &t;
		argsForBlock[numThreads - 1].noBlocks = blockCount - (numThreads - 1) * blocksPerThread;
		argsForBlock[numThreads - 1].startBlock = (numThreads - 1) * blocksPerThread;
		time = omp_get_wtime();
		
		for (int k = 0; k < numThreads; k++)
		{
			pthread_create(&threads_1[k], NULL, &blockTranspose, &argsForBlock[k]);
		}
		for (int k = 0; k < numThreads; k++)
		{
			pthread_join(threads_1[k], NULL);
		}
		time2 = omp_get_wtime() - time;
		times_block[testNo]=time2;
		printf("time elapsed (Block) : %f\n", ((float)time2) / 1);

		printf("\n");
		//save result to file
		fprintf(fp,"%d,%f,%f\n",matSizes[testNo],times_diag[testNo],times_block[testNo]);
		//free memory
		disposeRank2Tensor(&t);
	}

	fclose(fp);
	return 0;
}
