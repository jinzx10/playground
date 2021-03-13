#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <random>
#include <ctime>
#include <chrono>
#include <sstream>

using namespace std;
using ull = unsigned long long;
using iclock = std::chrono::high_resolution_clock;

int main(int, char** argv) {
	int num_procs;
	int id;
	double pi;
	iclock::time_point start;
	std::chrono::duration<double> dur;


	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);

	ull sz = 0;
	if (id == 0) {
		start = iclock::now();
		std::stringstream ss;
		ss << argv[1];
		ss >> sz;
	}
	::MPI_Bcast(&sz, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

	std::srand(id+std::time(nullptr));

	ull local_count = 0;
	ull tot_count = 0;
	double x = 0, y = 0;

	for (ull i = 0; i < sz/num_procs; ++i) {
		x = (double)std::rand() / RAND_MAX;
		y = (double)std::rand() / RAND_MAX;

		if (x*x+y*y < 1)
			++local_count;
	}

	::MPI_Reduce(&local_count, &tot_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if (id == 0) {
		pi = 4.0 * tot_count / sz;
		std::cout.precision(10);
		std::cout << "pi = " << pi << std::endl;
		dur = iclock::now() - start;
		std::cout << "time elapsed = " << dur.count() << std::endl; 
		std::cout << "num procs = " << num_procs << std::endl;
	}

	::MPI_Finalize();

	return 0;
}
