#include <type_traits>
#include <iostream>
#include <cmath>

template <typename F, typename T1, typename T2, typename = void>
struct is_callable : std::false_type {};

template <typename F, typename T1, typename T2>
struct is_callable<F, T1, T2, std::void_t<decltype( std::declval<F>()(std::declval<T1>(), std::declval<T2>()))>> : std::true_type {};

template <typename F, typename T, typename R, typename =void>
struct has_specific_signature : std::false_type {};

template <typename F, typename T, typename R>
struct has_specific_signature<F, T, R, std::void_t< std::is_same<R, decltype( std::declval<F>()(std::declval<T>()))> > > : std::true_type {};


struct foo 
{
	int operator()(){return 0;}
};

struct bar
{
	int operator()(double){return 0;}
};

struct tee
{
	int operator()(float, double){return 0;}
};

// don't work with functions!
double sum(double x, double y) {
	return x + y;
}

int multiply(int x, int y)
{
    return x * y;
}

int main() {

	std::cout << is_callable<foo, float, double>::value << std::endl;
	std::cout << is_callable<bar, float, double>::value << std::endl;
	std::cout << is_callable<tee, double, float>::value << std::endl;
	//std::cout << is_callable<multiply, float, double>::value << std::endl;
	std::cout << has_specific_signature<foo, double, int>::value << std::endl;
	std::cout << has_specific_signature<bar, double, int>::value << std::endl;

	return 0;
}
