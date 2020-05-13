#include "../utility/widgets.h"

int main() {

	std::cout << "start: " << trim("I don't know") << std::endl;
	std::cout << "start: " << trim(" I don't know  ") << std::endl;
	std::cout << "start: " << trim("		I don't know  ") << std::endl;
	std::cout << "start: " << trim("	\tI don't know  ", " ") << std::endl;
	std::cout << trim(" \tI don't know  ", " ") << std::endl;
	std::cout << trim(" \tI don't know  ", " ") << std::endl;
	std::cout << "\tI don't know" << std::endl;

	
	std::string str = trim(" \tI don't know  ", " ");

	int n = str.length();
	char arr[n+1];
	std::strcpy(arr, str.c_str());
	char* ptr = &arr[0];

	while (*ptr) {
		switch (*ptr) {
			case '\t': printf("\\t"); break;
			case '\n': printf("\\n"); break;
			default: printf("%c", *ptr);
		}
		++ptr;
	}
	std::cout << std::endl;

	return 0;
}
