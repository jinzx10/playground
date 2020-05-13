#include "../utility/widgets.h"

int main() {

	std::string str = "good day";
	std::string str2 = "   good day";
	std::string pre = "good";
	std::string pre2 = "bad";

	std::cout << std::boolalpha << start_with(pre, str) << std::endl;
	std::cout << std::boolalpha << start_with(pre2, str) << std::endl;
	std::cout << std::boolalpha << start_with(pre, str2) << std::endl;
	std::cout << std::boolalpha << start_with(pre, str2, " ") << std::endl;
	return 0;
}
