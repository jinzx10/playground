#include <iostream>

template <typename F, typename ...Args>
using return_t = decltype( std::declval<F>()( std::declval<Args>()... ) );

template <typename F, typename ...Args>
static bool test(return_t<F, Args...>* ) {
	return true;
}

template <typename ...C>
static bool test(...) {
	return false;
}


struct foo
{
	void operator()(int) {};
};

struct bar
{
	int operator()(int) {return 1;};
};


int main() {

	std::cout << typeid( return_t<foo, int> ).name() << std::endl;
	std::cout << typeid( return_t<bar, int> ).name() << std::endl;

	std::cout << test<foo,int>(nullptr) << std::endl;
	std::cout << test<foo,int,int>(nullptr) << std::endl;

	return 0;
}
