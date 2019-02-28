#include <stdio.h>
#include <stdlib.h>
#include "rank2Tensor.h"
#include <time.h>

void swap(int* i1,int* i2)
{
	 int temp =*i2;
	 *i2=*i1;
	 *i1=temp;
	
}

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

void DiagTranspose( rank2Tensor* t){
	
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
	rank2Tensor t;

	t.rows=4;
	t.cols=4;
	initRank2Tensor(&t);

	displayRank2Tensor(&t);
	blockTranspose(&t);

	printf("\n");
	displayRank2Tensor(&t);

	disposeRank2Tensor(&t);

	return 0;
}
