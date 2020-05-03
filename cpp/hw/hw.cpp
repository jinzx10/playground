#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>
#include "../utility/widgets.h"
#include "../utility/arma_helper.h"

using namespace arma;
using namespace std;

int main(int, char**argv) {

	std::string str = "  good day";

	auto start = str.find_first_not_of(" \t");

	str.erase(0, start);
	std::cout << str << std::endl;



    return 0;
}
