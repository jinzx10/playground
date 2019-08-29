#include <sstream>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
	
	if (argc < 3) {
		cerr << "must have 2 int arguments" << endl;
		return -1;
	}

	stringstream ss;
	ss << argv[1] << ' ' << argv[2];

	int a,b;

	ss >> a >> b;

	cout << "a = " << a << endl 
		<< "b = " << b << endl;


	return 0;
}
