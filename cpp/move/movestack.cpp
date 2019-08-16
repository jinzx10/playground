#include <iostream>
#include <array>

int main() {
	constexpr size_t sz = 3;
	std::array<double, sz> a = {5,3,1};
	std::array<double, sz> b = {1,2,3};

	std::cout << "a before move: " << &a << std::endl;
	std::cout << "b before move: " << &b << std::endl;

	std::array<double, sz> tmp = std::move(a);
	std::cout << "a after move: " << &a << std::endl;
	std::cout << "tmp before move: " << &tmp << std::endl;

	a = std::move(b);
	std::cout << "b after move: " << &b << std::endl;
	std::cout << "a after assign: " << &a << std::endl;

	b = std::move(tmp);
	std::cout << "b after assign: " << &b << std::endl;
	std::cout << "tmp after move: " << &tmp << std::endl;

	return 0;
}
