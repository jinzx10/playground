#include <iostream>
#include <mpi.h>
#include <chrono>
#include <sstream>
#include <cmath>

using namespace std;
using iclock = std::chrono::high_resolution_clock;
using ull = unsigned long long int;

int main(int, char** argv) {
	int nprocs;
	int id;
	double pi;
	iclock::time_point start;
	std::chrono::duration<double> dur;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);

	ull nbins_glb = 0;
	if (id == 0) {
		start = iclock::now();
		std::stringstream ss;
		ss << argv[1];
		ss >> nbins_glb;
	}
	::MPI_Bcast(&nbins_glb, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    double dx;
    dx = 1.0 / nbins_glb;

    ull rem = nbins_glb % nprocs;
    ull quo = nbins_glb / nprocs;
    ull nbins_loc = quo + ( id < rem ? 1 : 0 );
    ull ibin_start = id * quo + ( id < rem ? id : rem );

    double pi_loc = 0.0;
	for (ull ibin = ibin_start; ibin < ibin_start + nbins_loc; ++ibin) {
        pi_loc += 1.0 / ( 1.0 + std::pow(ibin*dx,2) ) * dx;
	}

	::MPI_Reduce(&pi_loc, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (id == 0) {
		pi *= 4.0;
		std::cout.precision(16);
		std::cout << "pi = " << pi << std::endl;
		dur = iclock::now() - start;
		std::cout << "time elapsed = " << dur.count() << std::endl; 
		std::cout << "num procs = " << nprocs << std::endl;
	}

	::MPI_Finalize();

	return 0;
}
