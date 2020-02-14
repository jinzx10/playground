#include <iostream>
#include <complex>
#include <armadillo>
#include <string>


int main() {

	std::string str = "";
	if ( str.length() )
		std::cout << "length true" << std::endl;
	std::cout << str.length() << std::endl;;

	if ( str.size() )
		std::cout << "size true" << std::endl;
	std::cout << str.size() << std::endl;;

	if ( str.empty())
		std::cout << "empty true" << std::endl;
	std::cout << str.empty() << std::endl;;

	return 0;
}
