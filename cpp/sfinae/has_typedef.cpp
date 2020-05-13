#include <iostream>
#include <type_traits>

template<typename ...>
using void_t = void;

template <typename T, typename = void>
struct has_typedef_foo : std::false_type {};

template <typename T>
struct has_typedef_foo<T, void_t<typename T::foo>> : std::true_type {};

class Foo {typedef int foo;};
class Bar {public: typedef int foo;};
class Tee {};

int main() {

	std::cout << has_typedef_foo<Foo>::value << std::endl;
	std::cout << has_typedef_foo<Bar>::value << std::endl;
	std::cout << has_typedef_foo<Tee>::value << std::endl;
	std::cout << has_typedef_foo<double>::value << std::endl;

	return 0;
}
