#include "../utility/widgets.h"


int main() {
	Parser p({"size", "value", "dir", "age", "name", "job"});
	p.parse("input.txt");

	int sz;
	double val;
	std::string dir;
	unsigned int age;
	std::string name;
	bool job;

	p.pour(sz, val, dir, age, name, job);

	std::cout << "size = " << sz << std::endl;
	std::cout << "value = " << val << std::endl;
	std::cout << "dir = " << dir << std::endl;
	std::cout << "age = " << age << std::endl;
	std::cout << "name = " << name << std::endl;
	std::cout << std::boolalpha << "job = " << job << std::endl;

	return 0;
}

