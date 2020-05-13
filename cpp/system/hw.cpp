#include <cstdlib>
#include <string>
#include <stdlib.h>
#include <iostream>

using namespace std;

int main() {

	string path = "~/playground/cpp/hw/hw.cpp";
	string real = realpath(path.c_str(), nullptr);

	cout << real << endl;

	return 0;
}
