#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

struct P
{
	template<typename T>
	void add(std::string const& key) {
		keys.push_back(key);
		vals.push_back(new T);
	}

	vector<std::string> keys;
	vector<void*> vals;
	
};

int main() {

	vector<void*> vv;
	vv.push_back(new int);
	vv.push_back(new double);

	*static_cast<int*>(vv[0]) = 34;
	*static_cast<double*>(vv[1]) = 3.14;

	std::cout << *static_cast<int*>(vv[0]) << std::endl;
	std::cout << *static_cast<double*>(vv[1]) << std::endl;

	

    return 0;
}
