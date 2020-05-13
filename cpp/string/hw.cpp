#include <string>
#include <iostream>

using namespace std;

int main() {

	string str = "dgda";
	cout << str.substr(str.size()-2) << endl;
	cout << str.substr(str.size()-1) << endl;
	cout << str.substr(str.size()-0) << endl;
	cout << str.substr(str.size()+1) << endl;
	return 0;
}
