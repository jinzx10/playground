#include <cstdio>
#include <mpi.h>
#include <unistd.h>

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

	void blacs_pinfo_(int *myid, int *nprocs);
	void blacs_get_(int* icontxt, int* what, int *val);
	void blacs_gridinit_( int *ictxt, const char *order, const int *nprow, const int *npcol );
	void blacs_gridinfo_( const int *ictxt, int *nprow, int *npcol, int *myprow, int *mypcol );

	int numroc_( const int *n, const int *nb, const int *iproc, const int *srcproc, const int *nprocs );
	void descinit_( 
		int *desc, 
		const int *m, const int *n, const int *mb, const int *nb, const int *irsrc, const int *icsrc, 
		const int *ictxt, const int *lld, int *info);

}

int main() {
    MPI_Init(nullptr, nullptr);

    int nprocs, rank;
    int group_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int ngrid = 4;

    MPI_Comm comm_sub;
    //MPI_Comm_split(MPI_COMM_WORLD, rank / ngrid, rank, &comm_sub);
    MPI_Comm_split(MPI_COMM_WORLD, (rank % 2) ? MPI_UNDEFINED : (rank / ngrid), rank, &comm_sub);
    if (rank % 2 == 0) {
        MPI_Comm_rank(comm_sub, &group_rank);
    }

    for (int i = 0; i < nprocs; i++) {
        if (i == rank) {
            printf("global rank = %d    group_rank = %d\n", rank, group_rank);
        }
        usleep(10000);
    }

    int np, id;
    blacs_pinfo_(&id, &np);

    //int ctxt = 0;
    char layout[] = "Row";

    int ctxt = 0;
    //MPI_Fint fcomm = MPI_Comm_c2f(comm_sub);
    //Cblacs_get(0, 0, &ctxt);
    ctxt = Csys2blacs_handle(comm_sub);
    for (int i = 0; i < nprocs; i++) {
        if (i == rank) {
            printf("(before init) MPI rank=%d/%d    BLACS id=%d/%d    ctxt = %d\n",
                    rank, nprocs, id, np, ctxt);
        }
        usleep(10000);
    }

    int nrow = 2, ncol = 2;
    Cblacs_gridinit(&ctxt, layout, nrow, ncol);

    int irow, icol;
    //Cblacs_pcoord(ctxt, id, &irow, &icol);
    Cblacs_gridinfo(ctxt, &nrow, &ncol, &irow, &icol);

    for (int i = 0; i < nprocs; i++) {
        if (i == rank) {
            printf("global rank=%d/%d    local rank = %d    BLACS id=%d/%d    (%d,%d)    ctxt = %d\n",
                    rank, nprocs, group_rank, id, np, irow, icol, ctxt);
        }
        usleep(10000);
    }

    MPI_Finalize();
}
