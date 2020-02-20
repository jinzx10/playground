#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

void find(std::vector<std::string> const& vs, std::string& str, unsigned int& type) {
	for (type = 0; type != vs.size(); ++type) {
		auto pos = str.find(vs[type]);
		if (pos != std::string::npos) {
			str.erase(pos, vs[type].length());
			break;
		}
	}
}

int main() {
	std::fstream fs("input.txt");
	std::vector<std::string> keys = {"size", "value", "dir"};
	int sz = 0;
	double val = 0;
	std::string dir;


	std::string str;
	unsigned int itype;
	std::stringstream ss;

	while(std::getline(fs, str)) {
		find(keys, str, itype); 
		std::cout << "itype = " << itype << "   str = " << str << std::endl;
		ss << str;
		switch (itype) {
			case 0: ss >> sz; 
					break;
			case 1: ss >> val;
					break;
			case 2: ss >> dir;
		}
		ss.str("");
		ss.clear();
	}
	std::cout << "size = " << sz << std::endl;
	std::cout << "value = " << val << std::endl;
	std::cout << "dir = " << dir << std::endl;

	return 0;
}
