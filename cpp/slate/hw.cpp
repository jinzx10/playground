#include "slate/Matrix.hh"
#include <slate/slate.hh>
#include <mpi.h>

int main() {

    MPI_Init(NULL, NULL);
    slate::Matrix<double> A( 3, 3, 1 , 1, 1, MPI_COMM_WORLD );

    MPI_Finalize();

}
