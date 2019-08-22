#include <vector>
#include <iostream>
#include <array>

int main() {
	// std::array
	constexpr size_t sz = 3;
	std::array<double, sz> arr1 = {5,3,1};
	std::cout << "arr1 before move: " << &arr1[0] << std::endl;
	std::array<double, sz> arr2 = std::move(arr1);
	std::cout << "arr1 after move: " << &arr1[0] << std::endl;
	std::cout << "arr2 after move construct: " << &arr2[0] << std::endl;
	std::cout << "contents of arr1: " << std::endl;
	for (auto& c : arr1)
		std::cout << c << std::endl;

	// std::vector
	// move construction
	std::vector<int> vec1 = {1,2,3};
	std::cout << "vec1 before move: " << &vec1[0] << std::endl;
	std::vector<int> vec2 = std::move(vec1);
	std::cout << "vec1 after move: " << &vec1[0] << std::endl;
	std::cout << "vev2 after move construct: " << &vec2[0] << std::endl;

	std::cout << "contents of vec1: " << std::endl;
	for (auto& c : vec1)
		std::cout << c << std::endl;

	// move copy assignment
	std::vector<int> vec3(3,0);
	std::cout << "vec3 before move: " << &vec3[0] << std::endl;
	vec3 = std::move(vec2);
	std::cout << "vec3 after move copy: " << &vec3[0] << std::endl;
	std::cout << "vec2 after move copy: " << &vec2[0] << std::endl;


	return 0;
}
