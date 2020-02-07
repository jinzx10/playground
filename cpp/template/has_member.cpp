#include <type_traits>
#include <utility>
#include <iostream>
#include <functional>

template <typename T, typename = void>
struct has_member_foo_v1 : std::false_type {};

template <typename T>
struct has_member_foo_v1<T, decltype(std::declval<T>().foo, void())> : std::true_type {}; // pass in gcc-8.2 or higher
//struct has_member_foo_v1<T, decltype(T::foo, void())> : std::true_type {}; // always private member SFINAE error

class Foo {int foo;};
class Bar {public: int foo;};
class Tee {};
class Kai {int foo();};
class Hou {public: int foo();};
class Moo {static int foo();};
class Mew {public: static int foo();};

int main() {

	std::cout << has_member_foo_v1<Foo>::value << std::endl;
	std::cout << has_member_foo_v1<Bar>::value << std::endl;
	std::cout << has_member_foo_v1<Tee>::value << std::endl;
	std::cout << has_member_foo_v1<Kai>::value << std::endl;
	std::cout << has_member_foo_v1<Hou>::value << std::endl;
	std::cout << has_member_foo_v1<Moo>::value << std::endl;
	std::cout << has_member_foo_v1<Mew>::value << std::endl;

	return 0;
}
