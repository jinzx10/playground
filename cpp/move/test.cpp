#include <vector>
#include <iostream>
#include <array>
#include <string>

template <typename V>
void print(V const& v, std::string const& name) {
	std::cout << "content of " << name << "   : ";
	for (auto e : v)
		std::cout << e << ' ';
	std::cout << std::endl;
}

template <typename V>
void info(V const& v, std::string const& name) {
	print<V>(v, name);
	std::cout << "address of " << name << "[0]: " << &v[0] << std::endl;
	std::cout << "address of " << name << "   : " << &v << std::endl;
}

int main() {
	// std::array
	constexpr size_t sz = 3;

	std::array<double, sz> arr1 = {5,3,1};
	info(arr1, "arr1");
	std::cout << "move-construct arr2 from arr1" << std::endl;
	std::array<double, sz> arr2 = std::move(arr1);
	info(arr1, "arr1");
	info(arr2, "arr2");

	std::cout << std::endl;


	// std::vector
	std::vector<int> vec1 = {1,2,3};
	info(vec1, "vec1");
	std::cout << "move-construct vec2 from vec1" << std::endl;
	std::vector<int> vec2 = std::move(vec1);
	info(vec1, "vec1");
	info(vec2, "vec2");

	std::cout << std::endl;


	// std::vector assignment
	std::vector<int> vec3(3,0);
	info(vec2, "vec2");
	info(vec3, "vec3");
	std::cout << "move-assign vec2 to vec3" << std::endl;
	vec3 = std::move(vec2);
	info(vec2, "vec2");
	info(vec3, "vec3");

	std::cout << std::endl;

	return 0;
}
