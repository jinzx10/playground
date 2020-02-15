#include <iostream>
#include <type_traits>
#include <functional>
#include <cmath>

/* get_return<Args...>(func)::type
 */

template <typename ...Args>
struct get_return
{

	//template <typename R>
	//static const std::function<R(Args...)> get_type(R(*F)(Args...));

	template <typename R>
	static void get_type(std::function<R(Args...)>);
	

};

template<int, typename ...>
void test(...) {}

template <int N = 5, typename F>
typename std::enable_if< N >= 0, void>::type test(F f) {
	std::cout << N << std::endl;
	test<N-1,F>(f);
}



int sum(int x, int y) {
	return x+y;
}

double sum(double x, double y) {
	return x+y;
}

int main() {

	//std::cout << typeid( get_return<double,double>::get_type(std::pow) ).name() << std::endl;
	test(1.0);

	return 0;
}
