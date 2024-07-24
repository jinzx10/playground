#include "blacs_connector.h"

#include <unistd.h>
#include <complex>

int main() {

    MPI_Init(NULL, NULL);

    int id = 0;
    int nprocs = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    int blacs_ctxt = Csys2blacs_handle(MPI_COMM_WORLD);
    char order = 'C';
    int nprow = 2;
    int npcol = 3;
    Cblacs_gridinit(&blacs_ctxt, &order, nprow, npcol);

    int iprow = 0, ipcol = 0;
    Cblacs_gridinfo(blacs_ctxt, &nprow, &npcol, &iprow, &ipcol);

    for (int i = 0; i < nprocs; ++i) {
        if (i == id) {
            printf("Rank = %i    pcoord = (%i,%i)\n", id, iprow, ipcol);
        }
        usleep(10000);
    }

    int data[3] = {};
    std::complex<double> zdata[3] = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}};
    char scope = 'A';
    char top = ' ';
    if (iprow == 1 && ipcol == 2) {
        data[0] = 9;
        data[1] = 8;
        data[2] = 7;
        Cigebs2d(blacs_ctxt, &scope, &top, 3, 1, data, 3);
        Czgebs2d(blacs_ctxt, &scope, &top, 3, 1, zdata, 3);
    } else {
        Cigebr2d(blacs_ctxt, &scope, &top, 3, 1, data, 3, 1, 2);
        Czgebr2d(blacs_ctxt, &scope, &top, 3, 1, zdata, 3, 1, 2);
    }

    for (int i = 0; i < nprocs; ++i) {
        if (i == id) {
            printf("Rank = %i    data = %i %i %i    zdata = (%6.1f,%6.1f) (%6.1f,%6.1f) (%6.1f,%6.1f)\n",
                    id, data[0], data[1], data[2],
                    zdata[0].real(), zdata[0].imag(),
                    zdata[1].real(), zdata[1].imag(),
                    zdata[2].real(), zdata[2].imag());
        }
        usleep(10000);
    }




    MPI_Finalize();

    return 0;
}
