rm -f *.o
rm -f *.csv
echo "compiling"
gcc Omp.c -o omp.o -fopenmp
gcc Pthreads.c -o pt.o -lpthread -fopenmp
gcc Serial.c -o ser.o -fopenmp
echo "done compiling"
echo "OMP Version"
./omp.o
echo "PThreads Version"
./pt.o
echo "Serial Version"
./ser.o