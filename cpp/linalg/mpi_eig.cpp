#include <mpi.h>
#include <armadillo>
#include <chrono>
#include <sstream>

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main(int, char**argv) {

	int id, nprocs;
	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int sz = 0, nt = 0;
	std::stringstream ss;

	if (id == 0) {
		ss << argv[1];
		ss >> sz;
		ss.clear();
		ss.str("");

		ss << argv[2];
		ss >> nt;
		ss.clear();
		ss.str("");
	}

	MPI_Bcast(&sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nt, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	mat a = randu(sz, sz);
	a += a.t();

	mat evec(sz, sz);
	vec eval(sz);

	iclock::time_point start = iclock::now();

	for (int i = 0; i != nt; ++i) {
		eig_sym(eval, evec, a);
	}

	std::chrono::duration<double> dur = iclock::now() - start;

	double t = dur.count() / nt;
	vec durs(nprocs);
	MPI_Gather(&t, 1, MPI_DOUBLE, durs.memptr(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (id == 0) {
		std::cout << "average time elapsed of each proc:" << std::endl;
		durs.print();
	}

	MPI_Finalize();

	return 0;
}
