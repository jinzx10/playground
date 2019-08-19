#include <iostream>
#include <tuple>

int main() {
	
	double a = 3, b = 4;
	double c = 1, d = 2;
	std::tie(a,b) = std::make_tuple(c,d);

	std::cout << a << b << std::endl;


	return 0;
}
