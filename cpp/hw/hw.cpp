#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <armadillo>
#include <type_traits>
#include "../utility/widgets.h"

using namespace std;

int main() {

	Stopwatch sw;
	auto sleep = [] (double x) {
		std::string cmd = "sleep " + std::to_string(x);
		std::system(cmd.c_str());
	};

	sw.run();

	sleep(0.1);

	sw.report(); // 0.1

	sw.run(5);
	sw.pause();

	sleep(0.2);
	sw.run();
	sw.report(5); // 0.2
	sw.report(3, "bad"); // no
	sw.report(0, "?"); // 0.1
	sw.reset(5);
	sw.report(5); // 0
	sw.run(5);
	sleep(0.2);
	sw.report(5, "good"); // 0.2

	

    return 0;
}
