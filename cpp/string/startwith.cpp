#include "../utility/widgets.h"

int main(int, char** argv) {

	std::string str, pre;
	readargs(argv, pre, str);

	std::cout << std::boolalpha << start_with(pre, str) << std::endl;
	return 0;
}
