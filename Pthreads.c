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

	int blockCount = blocks->noBlocks;
	int startBlock = blocks->startBlock;
	int endBlock=blockCount+startBlock-1;
	rank2Tensor* t = blocks->srcMat;
	int startRow = (int) ceil( (-1 +sqrt(1+startBlock*4))/2);
	int startRowStart=((startRow+1)*(startRow+2))/2 - (startRow+1); //blocks per row= row index +1
	int startCol = (startBlock-startRowStart)*2;

	int endRow = (int) ceil((-1 +sqrt(1+endBlock*4))/2);
	int endRowStart = ((endRow+1)*(endRow+2))/2 - (endRow+1);
	int endCol = (endBlock-endRowStart)*2;

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

	
	int numThreads=8;
	rank2Tensor t;


	int N_0=10;
	t.rows=N_0;
	t.cols=N_0;
	initRank2Tensor(&t);

	forDiag argsForDiag[numThreads];

	pthread_t threads[numThreads];
	displayRank2Tensor(&t);

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


	printf("\n");
	displayRank2Tensor(&t);

	double time = omp_get_wtime();
	double time2 = omp_get_wtime() - time;
	printf("time elapsed (Diag) : %f\n",((float)time2)/1);

//////////////////
	int blockCount = t.rows/2;
	blockCount=blockCount*(blockCount+1)/2;

	forBlock argsForBlock[numThreads];
	pthread_t threads_1[numThreads];
	if(blockCount<numThreads){
		numThreads=blockCount;
	}

	int blocksPerThread=blockCount/numThreads;
	argsForBlock[0].srcMat=&t;
	argsForBlock[0].noBlocks=blocksPerThread;
	argsForBlock[0].startBlock=0;
	//printf("%d %d %d\n",blockCount, numThreads, blocksPerThread);


	for (int i = 1; i < numThreads-1; ++i)
	{
		//printf("%d",i);
		argsForBlock[i].srcMat=&t;
		argsForBlock[i].noBlocks=blocksPerThread;
		argsForBlock[i].startBlock=i*blocksPerThread;

	}
	argsForBlock[numThreads-1].srcMat=&t;
	argsForBlock[numThreads-1].noBlocks=blockCount-(numThreads-1)*blocksPerThread;
	argsForBlock[numThreads-1].startBlock=(numThreads-1)*blocksPerThread;
	for (int k = 0; k<numThreads; k++)
	{
		//printf("%d\n",k);
		pthread_create(&threads_1[k],NULL,&blockTranspose,&argsForBlock[k]);
		//printf("%d\n",k);

	}
	for (int k = 0; k<numThreads; k++)
	{		
		//printf("%d\n",k);
		pthread_join(threads_1[k],NULL);
	}

	
	printf("\n");
	displayRank2Tensor(&t);

/////////////////
	disposeRank2Tensor(&t);

	return 0;
}
