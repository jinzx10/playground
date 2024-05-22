#ifndef SCALAPACK_H
#define SCALAPACK_H

#include <mpi.h>

extern "C"
{
    int Csys2blacs_handle(MPI_Comm SysCtxt);
    int Cblacs_pnum(int icontxt, int prow, int pcol);
	void Cblacs_pinfo(int *myid, int *nprocs);
	void Cblacs_get(int icontxt, int what, int *val);
	void Cblacs_gridmap(int* icontxt, int *usermap, int ldumap, int nprow, int npcol);
	void Cblacs_gridinfo(int icontxt, int* nprow, int *npcol, int *myprow, int *mypcol);
    void Cblacs_gridinit(int* icontxt, char* layout, int nprow, int npcol);
    void Cblacs_pcoord(int icontxt, int pnum, int *prow, int *pcol);
	void Cblacs_exit(int icontxt);

	int numroc_(const int *n, const int *nb, const int *iproc, const int *srcproc, const int *nprocs);
	void descinit_(int *desc, const int *m, const int *n, const int *mb, const int *nb, const int *irsrc, const int *icsrc, const int *ictxt, const int *lld, int *info);

    void Cpdgemr2d(int m, int n, double* A, int IA, int JA, int *descA, double* B, int IB, int JB, int *descB, int gcontext);
}

#endif
