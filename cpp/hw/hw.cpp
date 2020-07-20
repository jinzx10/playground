#include <chrono>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>
#include <cassert>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int, char**argv) {
	
	char str[] = "good day";
	cout << typeid(str).name() << std::endl;
	return 0;
}
