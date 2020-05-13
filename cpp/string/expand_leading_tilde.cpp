#include <string>
#include <iostream>
#include "../utility/widgets.h"
#include <cstring>

using namespace std;

int main(int, char** argv) {

	string str;
	readargs(argv, str);

	cout << ::expand_leading_tilde(str) << endl;

	return 0;
}
