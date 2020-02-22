#include <type_traits>
#include <sstream>
#include <iostream>
#include <string>
#include "../utility/widgets.h"

using namespace std;

int main(int argc, char** argv) {
	
	std::string str;
	int i;
	std::stringstream ss;

	ss << argv[1];

	std::cout << "ss.str() = " << ss.str() << std::endl;

	string sa,sb;
	if (std::is_same<std::string, decltype(sa)>::value) {
		std::getline(ss, sa);
	}
	std::cout << "sa = " << sa << std::endl;


	/*
	if (!ss.eof()) {
		std::cout << "something left" << std::endl;
	}

	ss >> sb;
	std::cout << "sb = " << sb << std::endl;
	std::cout << ss.str() << std::endl;

	*/

	if (ss.eof()) {
		std::cout << "nothing left" << std::endl;
	}


	return 0;
}
