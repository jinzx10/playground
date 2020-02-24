#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <armadillo>
#include <type_traits>

using namespace std;

int main() {

	std::cout << std::is_trivial<std::string>::value << std::endl;

	std::string a = "good";
	std::cout << a.size() << std::endl;
	std::cout << a.data() << std::endl;

	arma::mat b = arma::ones(3,5);

	auto sz = arma::size(b);
	arma::cube v = arma::ones(4,5,6);
	auto sz2 = arma::size(v);

	

    return 0;
}
