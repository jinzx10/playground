#include <iostream>


template <typename T>
int test(typename T::foobar*) {
	return 1;
}


template <typename ...>
int test(...) {
	return 0;
}

struct foo
{
	typedef int foobar;
};

int main() {
	std::cout << test<int>(nullptr) << std::endl;
	std::cout << test<foo>(nullptr) << std::endl;

	return 0;
}
