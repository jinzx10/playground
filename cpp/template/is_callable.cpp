#include <type_traits>
#include <iostream>
#include <cmath>

template <typename F, typename T1, typename T2, typename = void>
struct is_callable : std::false_type {};

template <typename F, typename T1, typename T2>
struct is_callable<F, T1, T2, std::void_t<decltype( std::declval<F>()(std::declval<T1>(), std::declval<T2>()))>> : std::true_type {};

//struct foo 
//{
//	int operator()(){return 0;}
//};

struct bar
{
	int operator()(double, float){return 0;}
};

struct tee
{
	int operator()(float, double){return 0;}
};

double sum(double x, double y) {
	return x+ y;
}

double multiply(double x, double y) {
	return x*y;
}

int multiply(int x, int y)
{
    return x * y;
}

template <class F>
void foo(int x, int y, F f)
{
	std::cout << f(x, y) << std::endl;
}


int main()
{
    foo(3, 4, multiply); // works

    return 0;
}
//int main() {
//
//	std::cout << is_callable<foo, float, double>::value << std::endl;
//	std::cout << is_callable<bar, float, double>::value << std::endl;
//	std::cout << is_callable<tee, float, double>::value << std::endl;
//	std::cout << is_callable<multiply, float, double>::value << std::endl;
//
//	return 0;
//}
