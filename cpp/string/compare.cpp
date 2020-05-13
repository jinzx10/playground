#include <iostream>
#include <string>
#include <cstring>
#include "../utility/widgets.h"

using namespace std;

void check(string const& a) {
	if (!a.compare("good")) { // return 0 if equal
		std::cout << "good" << std::endl;
	} else if (!a.compare("bad")) {
		std::cout << "bad" << std::endl;
	} else {
		std::cout << "others" << std::endl;
	}
}


int main(int, char** argv) {
	string s1;
	readargs(argv, s1);

	cout << s1.compare("~") << endl;
	cout << s1.compare("~/") << endl;
	

	return 0;
}
