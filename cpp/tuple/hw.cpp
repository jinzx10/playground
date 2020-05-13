#include <iostream>
#include <tuple>
#include <typeinfo>

template <typename ...Ts>
struct T
{
	std::tuple<Ts...> types;

};

int main() {
	
	T<int,double,char> t;

	std::tuple<int, double, char> tp = {3, 3.14, 'p'};

	std::cout << typeid( decltype(std::get<0>(t.types)) ).name() << std::endl;
	std::cout << typeid( decltype(std::get<2>(t.types)) ).name() << std::endl;


	int i = 0;
	double x = 0.0;
	char z = '\0';
	std::tie(i, x, z) = tp;


	return 0;
}
