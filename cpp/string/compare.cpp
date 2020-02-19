#include <iostream>
#include <string>

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


int main() {
	string s1 = "good";
	
	std::cout << (s1 == "good") << std::endl;
	std::cout << (s1 == "bad") << std::endl;

	check("good");
	check("bad");
	check("no");

	return 0;
}
