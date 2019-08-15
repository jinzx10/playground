#include <armadillo>
#include <iterator>
#include <array>
#include <vector>
#include <iostream>

template <typename ...>
using void_t = void;

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, void_t<decltype(std::begin(std::declval<T&>())), decltype(std::end(std::declval<T&>()))>> : std::true_type {};

struct Foo {};

int main() {

	std::cout << is_iterable<double>::value << std::endl;
	std::cout << is_iterable<Foo>::value << std::endl;
	std::cout << is_iterable<arma::vec>::value << std::endl;
	std::cout << is_iterable<std::vector<int>>::value << std::endl;
	std::cout << is_iterable<std::array<int,1>>::value << std::endl;
	std::cout << is_iterable<double[3]>::value << std::endl;

	return 0;
}

