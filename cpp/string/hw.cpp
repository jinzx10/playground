#include <string>
#include <iostream>

using namespace std;

int main() {

	string str = " dgda";
	auto start = str.find_first_not_of(" ");
	cout << start << endl;
	return 0;
}
