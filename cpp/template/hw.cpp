#include "aux.h"
#include <vector>
#include <array>
#include <armadillo>
#include <iostream>

int main() 
{
	std::array<int,3> ai = {1,2,3};
	std::vector<int> vi = {0, -1, -3};
	arma::vec av = {5,6,7};
	int* pi = &ai[0];

	std::cout << is_iterable<std::vector<int>>::value << std::endl;
	std::cout << is_iterable<std::array<int,3>>::value << std::endl;
	std::cout << is_iterable<arma::vec>::value << std::endl;
	std::cout << is_iterable<double>::value << std::endl;

	std::cout << std::is_lvalue_reference< decltype(std::declval<decltype( std::begin( std::declval<arma::vec&>() ) ) >() ) >::value 
		<< std::endl;
	std::cout << std::is_lvalue_reference< decltype(std::declval<decltype( std::begin( std::declval<arma::vec&>() ) )&>() ) >::value
		<< std::endl;


	return 0;
}
	
