#include "stopwatch.h"
#include <cstdlib>
#include <string>

int main() {
	std::string command = "sleep 1";
	Stopwatch sw;
	std::system(command.c_str());

	sw.report(); // 0
	sw.run();

	std::system(command.c_str());
	sw.report(); // 1

	sw.pause();
	std::system(command.c_str());

	sw.report(); // 1
	sw.run();
	std::system(command.c_str());
	sw.report(); // 2

	sw.reset();
	sw.report(); // 0
	std::system(command.c_str());
	sw.report(); // 0

	sw.run();
	std::system(command.c_str());
	sw.report(); // 1


	return 0;
}
