#include <iostream>

template <typename T>
struct has_typedef_foobar
{
	typedef char yes;
	typedef char no[2];

	template <typename C>
	static yes& test(typename C::foobar*);

	template <typename C>
	static no& test(...);

	static const bool value = sizeof(test<T>(nullptr)) == sizeof(yes);

};



struct foo
{
	typedef int foobar;
};

int main() 
{

	std::cout << has_typedef_foobar<int>::value << std::endl;
	std::cout << has_typedef_foobar<foo>::value << std::endl;


	return 0;
}
	
