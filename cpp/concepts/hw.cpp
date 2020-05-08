#include <iostream>
#include <concepts>
#include <type_traits>

template <typename ...>
using void_t = void;

template <typename T, typename = void>
struct has_member_tq : std::false_type {};

template <typename T>
struct has_member_tq<T, void_t<decltype(std::declval<T>().tq)>  > : std::true_type {};

template <typename T>
concept Tq = has_member_tq<T>::value;


template <Tq T>
void addone(T& t) {
	++t.tq;
	std::cout << t.tq << std::endl;
}

struct Test
{
	Test(double x) : tq(x) {}
	double tq;
};
	


int main() {

	Test t(5);
	addone(t);



	return 0;
}
