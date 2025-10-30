#include <iostream>
#include <cstddef>

struct S {
    char c[3];
	int i;
    double d;
};

int main()
{
	std::cout << offsetof(S, c) << std::endl
		<< offsetof(S,i) << std::endl
		<< offsetof(S,d) << std::endl
		<< std::endl;
	S s;
	std::cout << &s.c << std::endl;
	std::cout << &s.i << std::endl;
	std::cout << &s.d << std::endl;
}
