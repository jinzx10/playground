#include <type_traits>
#include <sstream>
#include <iostream>
#include <string>
#include "../utility/widgets.h"

using namespace std;

int main(int argc, char** argv) {
	
	double x;

	readargs(argv, x);

	std::cout << x << std::endl;


	return 0;
}
