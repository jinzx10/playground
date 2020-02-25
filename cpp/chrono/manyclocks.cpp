#include <chrono>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>

using iclock = std::chrono::high_resolution_clock;
using dur_t = std::chrono::duration<double>;
using tt = iclock::time_point;

int main() {

	auto sleep = [] (double dur) { 
		std::string cmd = "sleep " + std::to_string(dur);
		std::system(cmd.c_str()); 
	};

	
	// use std::vector
	// stupid! should be obsolete
	std::vector<tt> vt(1);
	std::vector<dur_t> vd(1);
	vt[0] = iclock::now();

	sleep(0.1);

	vd[0] = iclock::now() - vt[0];

	std::cout << vd[0].count() << std::endl;

	vt.push_back(iclock::now());

	sleep(0.2);

	vd.push_back(iclock::now() - vt[1]);
	std::cout << vd[1].count() << std::endl;

	// use std::map
	std::map<unsigned int, tt> ts;
	std::map<unsigned int, dur_t> ds;
	ts[0] = iclock::now();

	sleep(0.1);

	ds[0] = iclock::now() - ts[0];
	std::cout << ds[0].count() << std::endl;




	return 0;
}
