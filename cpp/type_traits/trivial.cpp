#include <type_traits>
#include <iostream>
#include <armadillo>
#include <vector>

int main() {

	std::cout << "double: " << std::is_trivial<double>::value << std::endl;
	std::cout << "int: " << std::is_trivial<int>::value << std::endl;
	std::cout << "char: " << std::is_trivial<char>::value << std::endl;
	std::cout << "vector<double>: " << std::is_trivial<std::vector<double>>::value << std::endl;
	std::cout << "vector<int>: " << std::is_trivial<std::vector<int>>::value << std::endl;
	std::cout << "vector<char>: " << std::is_trivial<std::vector<char>>::value << std::endl;
	std::cout << "mat: " << std::is_trivial<arma::mat>::value << std::endl;
	std::cout << "vec: " << std::is_trivial<arma::vec>::value << std::endl;
	std::cout << "umat: " << std::is_trivial<arma::umat>::value << std::endl;
	std::cout << "string: " << std::is_trivial<std::string>::value << std::endl;

	std::cout << "double: " << std::is_standard_layout<double>::value << std::endl;
	std::cout << "int: " << std::is_standard_layout<int>::value << std::endl;
	std::cout << "char: " << std::is_standard_layout<char>::value << std::endl;
	std::cout << "vector<double>: " << std::is_standard_layout<std::vector<double>>::value << std::endl;
	std::cout << "vector<int>: " << std::is_standard_layout<std::vector<int>>::value << std::endl;
	std::cout << "vector<char>: " << std::is_standard_layout<std::vector<char>>::value << std::endl;
	std::cout << "mat: " << std::is_standard_layout<arma::mat>::value << std::endl;
	std::cout << "vec: " << std::is_standard_layout<arma::vec>::value << std::endl;
	std::cout << "umat: " << std::is_standard_layout<arma::umat>::value << std::endl;
	std::cout << "string: " << std::is_standard_layout<std::string>::value << std::endl;

	return 0;
}
