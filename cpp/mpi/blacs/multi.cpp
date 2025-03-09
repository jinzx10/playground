#include <cstdio>
#include <mpi.h>
#include <unistd.h>
#include <cassert>

extern "C"
{
    int Csys2blacs_handle(MPI_Comm SysCtxt);
	void Cblacs_pinfo(int *myid, int *nprocs);
	void Cblacs_get(int icontxt, int what, int *val);
	void Cblacs_gridmap(int* icontxt, int *usermap, int ldumap, int nprow, int npcol);
	void Cblacs_gridinfo(int icontxt, int* nprow, int *npcol, int *myprow, int *mypcol);
    void Cblacs_gridinit(int* icontxt, char* layout, int nprow, int npcol);
    int Cblacs_pnum(int icontxt, int prow, int pcol);
    void Cblacs_pcoord(int icontxt, int pnum, int *prow, int *pcol);
	void Cblacs_exit(int icontxt);

	int numroc_(const int *n, const int *nb, const int *iproc, const int *srcproc, const int *nprocs);
	void descinit_(int *desc, const int *m, const int *n, const int *mb, const int *nb, const int *irsrc, const int *icsrc, const int *ictxt, const int *lld, int *info);

}

int main() {
    MPI_Init(nullptr, nullptr);

    /***************************************************************************
     *                  MPI rank and number of processes
     ***************************************************************************/
    int nprocs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /***************************************************************************
     *                  BLACS rank and number of processes
     ***************************************************************************/
    int np, id;
    Cblacs_pinfo(&id, &np);
    // this is not very useful because it gives the same rank as MPI
    // see https://netlib.org/scalapack/explore-html/d6/dd3/blacs__pinfo___8c_source.html

    /***************************************************************************
     *                      split the MPI communicator
     ***************************************************************************/
    // number of disjoint communicators
    // each of which will be used to create a BLACS context
    int n_sub = 4;
    int rank_sub = -1;
    int nprocs_sub = 0;
    MPI_Comm comm_sub;
    MPI_Comm_split(MPI_COMM_WORLD, rank / n_sub, rank, &comm_sub);
    MPI_Comm_rank(comm_sub, &rank_sub);
    MPI_Comm_size(comm_sub, &nprocs_sub);

    assert(n_sub <= nprocs);

    /***************************************************************************
     *                              ranks
     ***************************************************************************/
    for (int i = 0; i < nprocs; i++) {
        if (i == rank) {
            printf("MPI rank (world) = %i/%i    MPI rank (sub) = %i/%i    BLACS rank = %i\n",
                    rank, nprocs, rank_sub, nprocs_sub, id);
        }
        usleep(10000);
    }
    if (rank == 0) printf("\n");


    /***************************************************************************
     *                          system context
     ***************************************************************************/
    // MPI communicator to BLACS context (will be used in grid_init)
    int ctxt_sub = Csys2blacs_handle(comm_sub);
    int ctxt_world = Csys2blacs_handle(MPI_COMM_WORLD);

    if (rank == 0) printf("system ctxt id (i.e., MPI communicator index):\n");
    for (int i = 0; i < nprocs; i++) {
        if (i == rank) {
            printf("MPI rank (world) = %i    ctxt (world) = %i    ctxt (sub) = %i\n",
                    rank, ctxt_world, ctxt_sub);
        }
        usleep(10000);
    }
    if (rank == 0) printf("\n");

    /***************************************************************************
     *                          BLACS grid init
     ***************************************************************************/
    int nrow, ncol, irow, icol;
    char layout = 'r'; // 'c' or 'C' for column major; anything else yields row major

    // initialized a BLACS grid from MPI_COMM_WORLD
    nrow = 2, ncol = 3;
    Cblacs_gridinit(&ctxt_world, &layout, nrow, ncol);

    //Cblacs_pcoord(ctxt_world, id, &irow, &icol);
    Cblacs_gridinfo(ctxt_world, &nrow, &ncol, &irow, &icol);

    // NOTE: blacs_gridinfo is preferred over blacs_pcoord because it can handle
    // the case where some processes are not part of the grid, in which case irow
    // and icol are set to -1. For blacs_pcoord, error is thrown in this case.

    if (rank == 0) printf("%ix%i BLACS grid from MPI_COMM_WORLD:\n", nrow, ncol);
    for (int i = 0; i < nprocs; i++) {
        if (i == rank) {
            printf("MPI rank (world) = %i    pcoord (%i,%i)    BLACS ctxt = %i\n",
                    rank, irow, icol, ctxt_world);
        }
        usleep(10000);
    }
    if (rank == 0) printf("\n");

    /****************************************************************************/

    // initialize a BLACS grid from comm_sub
    ctxt_sub = Csys2blacs_handle(comm_sub);

    nrow = 1, ncol = 3;
    Cblacs_gridinit(&ctxt_sub, &layout, nrow, ncol);
    Cblacs_gridinfo(ctxt_sub, &nrow, &ncol, &irow, &icol);

    if (rank == 0) printf("%ix%i BLACS grid from comm_sub:\n", nrow, ncol);
    for (int i = 0; i < nprocs; i++) {
        if (i == rank) {
            printf("MPI rank (sub) = %i    pcoord (%i,%i)    BLACS ctxt = %i\n",
                    rank_sub, irow, icol, ctxt_sub);
        }
        usleep(10000);
    }
    if (rank == 0) printf("\n");

    /****************************************************************************/

    // initialize another BLACS grid from comm_sub
    ctxt_sub = Csys2blacs_handle(comm_sub);

    nrow = 2, ncol = 2;
    Cblacs_gridinit(&ctxt_sub, &layout, nrow, ncol);
    Cblacs_gridinfo(ctxt_sub, &nrow, &ncol, &irow, &icol);

    if (rank == 0) printf("%ix%i BLACS grid from comm_sub (Cblacs_gridinfo):\n", nrow, ncol);
    for (int i = 0; i < nprocs; i++) {
        if (i == rank) {
            printf("MPI rank (sub) = %i    pcoord (%i,%i)    BLACS ctxt = %i\n",
                    rank_sub, irow, icol, ctxt_sub);
        }
        usleep(10000);
    }
    if (rank == 0) printf("\n");


    // NOTE: the second argument passed to pcoord should be the rank within the
    // relevant communicator (which may not be the rank in MPI_COMM_WORLD)
    MPI_Comm_rank(comm_sub, &id);
    Cblacs_pcoord(ctxt_sub, id, &irow, &icol);
    if (rank == 0) printf("%ix%i BLACS grid from comm_sub (Cblacs_pcoord):\n", nrow, ncol);
    for (int i = 0; i < nprocs; i++) {
        if (i == rank) {
            printf("MPI rank (sub) = %i    pcoord (%i,%i)    BLACS ctxt = %i\n",
                    rank_sub, irow, icol, ctxt_sub);
        }
        usleep(10000);
    }
    if (rank == 0) printf("\n");




    MPI_Finalize();
}
