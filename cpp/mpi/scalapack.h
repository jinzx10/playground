#ifndef __SCALAPACK_H__
#define __SCALAPACK_H__

extern "C"
{
	// BLACS
	void Cblacs_pinfo(int*, int*);
	void Cblacs_get(int, int, int*);
	void Cblacs_barrier(int, char*);
	void Cblacs_gridinit(int* , char*, int, int);
	void Cblacs_gridinfo(int, int*, int*, int*, int*);
	void Cblacs_gridexit(int);
	void Cdgesd2d(int, int, int, double*, int, int, int);
	void Cdgerv2d(int, int, int, double*, int, int, int);

	// PBLAS
	void pdgemm_(char*, char*, int*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*, int*);

	// ScaLAPACK utilities
	int numroc_(int const*, int const*, int const*, int const*, int const*);
	void descinit_(int*, int const*, int const*, int*, int*, int*, int*, int*, int*, int*); 
}

#endif
