#include <stdio.h>
#include <stdlib.h>
#include "rank2Tensor.h"
#include <time.h>

void swap(int* i1,int* i2)
{
	printf("%i\n",*i1 );

	 int temp =*i2;
	 *i2=*i1;
	 *i1=temp;
	printf("%i\n\n", *i1 );
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


int main ()
{
	srand(time(NULL));
	rank2Tensor t;

	t.rows=3;
	t.cols=3;
	initRank2Tensor(&t);

	displayRank2Tensor(&t);
	naiveTranspose(&t);

	printf("\n");
	displayRank2Tensor(&t);

	disposeRank2Tensor(&t);

	return 0;
}