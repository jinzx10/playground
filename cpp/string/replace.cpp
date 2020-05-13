#include <string>
#include <iostream>
#include "../utility/widgets.h"
#include <cstring>

using namespace std;

int main(int, char** argv) {

	string str;
	readargs(argv, str);

	auto start = str.find_first_not_of(" \t");
	if (std::strncmp(str.substr(start, 2).c_str(), "~/", 2) == 0) {
		str.replace(start, 2, std::getenv("HOME")+std::string("/"));
	}

	cout << str << endl;

	return 0;
}
