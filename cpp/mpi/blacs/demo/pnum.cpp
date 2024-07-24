#include "blacs_connector.h"

#include <unistd.h>

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

    for (int i = 0; i < nprocs; ++i) {
        if (i == id) {
            for (int r = 0; r < nprow; ++r) {
                for (int c = 0; c < npcol; ++c) {
                    int pnum = Cblacs_pnum(blacs_ctxt, r, c);
                    printf("pcoord = (%i,%i)    pnum = %i\n", r, c, pnum);
                }
            }
            usleep(20000);
        }
    }

    MPI_Finalize();

    return 0;
}
