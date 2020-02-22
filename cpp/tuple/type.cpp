#include <type_traits>
#include <tuple>
#include <typeinfo>

int main() {

	std::tuple<int, char, double> types;

	std::cout << std::is_same<int, decltype()>::value << std::endl;

	return 0;
}
