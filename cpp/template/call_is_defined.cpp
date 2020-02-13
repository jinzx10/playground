#include <type_traits>
#include <iostream>
#include <typeinfo>


template <typename F, typename ...Args>
struct call_is_defined
{
    typedef char yes;
    typedef char no[2];

	template <typename F1, typename ...Args1>
	using return_t = decltype( std::declval<F1>()( std::declval<Args1>()... ) );

    template <typename F1, typename ...Args1>
    static yes& test(return_t<F1, Args1...>* );

    template <typename F1, typename ...Args1>
    static no& test(...);

    static const bool value = sizeof(test<F, Args...>(nullptr)) == sizeof(yes);

	template <bool cond, typename F1, typename ...Args1>
	struct try_return { using type = void; };
	
	template <typename F1, typename ...Args1>
	struct try_return<true, F1, Args1...> { using type = return_t<F1, Args...>; };

	using return_type = typename try_return<value, F, Args...>::type;

};

double sum(double x, double y) {
	return x+y;
}

double addone(double x) {
	return x+1;
}

int main() {

	std::cout << call_is_defined<decltype(sum), double, double>::value << std::endl;
	std::cout << typeid(call_is_defined<decltype(sum), double, double>::return_type).name()<< std::endl;

	std::cout << call_is_defined<decltype(addone), double, double>::value << std::endl;
	std::cout << typeid(call_is_defined<decltype(addone), double, double>::return_type).name()<< std::endl;

	return 0;
}
