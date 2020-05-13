#include <type_traits>
#include <iostream>
#include <typeinfo>


template <typename F, typename ...Args>
class is_valid_call
{
    typedef char yes;
    typedef char no[2];

	template <typename F1, typename ...Args1>
	using return_t = decltype( std::declval<F1>()( std::declval<Args1>()... ) );

    template <typename F1, typename ...Args1>
    static yes& test(return_t<F1, Args1...>* );

    template <typename F1, typename ...Args1>
    static no& test(...);

	template <bool cond, typename F1, typename ...Args1>
	struct try_return { using type = void; };
	
	template <typename F1, typename ...Args1>
	struct try_return<true, F1, Args1...> { using type = return_t<F1, Args...>; };


	public:

    static const bool value = sizeof(test<F, Args...>(nullptr)) == sizeof(yes);
	using return_type = typename try_return<value, F, Args...>::type;

};

double sum(double x, double y) {
	return x+y;
}

double addone(double x) {
	return x+1;
}

int main() {

	std::cout << is_valid_call<decltype(sum), double, double>::value << std::endl;
	std::cout << typeid(is_valid_call<decltype(sum), double, double>::return_type).name()<< std::endl;

	std::cout << is_valid_call<decltype(addone), double, double>::value << std::endl;
	std::cout << typeid(is_valid_call<decltype(addone), double, double>::return_type).name()<< std::endl;

	return 0;
}
