#include <iostream>
#include <type_traits>

template <typename ...Ts, typename std::enable_if<sizeof...(Ts) >= 3, int>::type = 0>
void foo(Ts ... args) {
	std::cout << "more than 3" << std::endl;
}


template <typename ...Ts, typename std::enable_if<sizeof...(Ts) < 3, int>::type = 0>
void foo(Ts ... args) {
	std::cout << "less than 3" << std::endl;
}


int main() {

	foo(3,7.7,5);
	foo('g');

	return 0;
}

