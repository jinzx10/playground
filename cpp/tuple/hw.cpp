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

	std::cout << typeid( decltype(std::get<0>(t.types)) ).name() << std::endl;
	std::cout << typeid( decltype(std::get<2>(t.types)) ).name() << std::endl;



	return 0;
}
