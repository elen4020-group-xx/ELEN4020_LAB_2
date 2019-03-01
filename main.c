#include <stdio.h>
#include <stdlib.h>
#include "rank2Tensor.h"
#include <time.h>
#include "omp.h"

void swap(int* i1,int* i2)
{
	 int temp =*i2;
	 *i2=*i1;
	 *i1=temp;
	
}
void naiveTranspose(rank2Tensor* t)
{
	#pragma omp parallel for
 	for(int i=0; i<t->rows; i++)
 	{	
 		#pragma omp parallel for
 		for(int j=0; j<i; j++)
 		{
 			swap(&(t->matrix[i][j]),&(t->matrix[j][i]));
 		}

 	}

}

void DiagTranspose(rank2Tensor* t)
{
	#pragma omp parallel for	
	for(int i=0;i<t->rows;i++)
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


		#pragma omp parallel for
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

	unsigned long time = clock();
	

	rank2Tensor t;

	t.rows=81928*2;
	t.cols=8192*2;
	initRank2Tensor(&t);

	//displayRank2Tensor(&t);
	naiveTranspose(&t);

	unsigned long time2 = clock() - time;
	printf("time elapsed (Naive) : %f\n",((float)time2)/CLOCKS_PER_SEC);


	time = clock();
	DiagTranspose(&t);	

	time2 = clock() - time;
	printf("time elapsed diagonal: %f\n",((float)time2)/CLOCKS_PER_SEC);



	time = clock();
	blockTranspose(&t);	

	time2 = clock() - time;
	printf("time elapsed Block: %f\n",((float)time2)/CLOCKS_PER_SEC);

	//printf("\n");
	//displayRank2Tensor(&t);

	disposeRank2Tensor(&t);

	return 0;
}
