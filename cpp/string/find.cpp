#include <iostream>
#include <string>

using namespace std;

int main() {

	string str = " good day! ";
	auto start = str.find_first_not_of(" ");
	auto end = str.find_last_not_of(" ");

	cout << start << endl << end << endl;

	return 0;
}
