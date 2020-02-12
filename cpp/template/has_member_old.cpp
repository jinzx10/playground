#include <iostream>

template <typename T>
struct has_member_tq
{
	typedef char yes;
	typedef char no[2];

	template <typename C>
	static yes& test(decltype(std::declval<C>().tq)*);

	template <typename C>
	static no& test(...);

	static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);

};



struct foo
{
	typedef int foobar;
};

struct bar
{
	int tq;
};

int main() 
{

	std::cout << has_member_tq<int>::value << std::endl;
	std::cout << has_member_tq<foo>::value << std::endl;
	std::cout << has_member_tq<bar>::value << std::endl;


	return 0;
}
	

