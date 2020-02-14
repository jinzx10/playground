#include <iostream>

template <int (*F)(int,int)>
void print(int x, int y) {
	std::cout << F(x,y) << std::endl;
}

template <typename R, typename ...Args>
struct test
{
	using F = R(*)(Args...);

};


int add(int x, int y) {
	return x+y;
}


int main() {

	print<add>(3,5);

	return 0;
}
