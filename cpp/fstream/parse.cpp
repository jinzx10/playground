#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include "../utility/widgets.h"

int main() {
	Parser p ({"name", "age", "job", "birthday"});
	p.parse("input.txt");

	std::string name,
				job,
				location,
				birth_city,
				birthday;
	unsigned int age;

	p.pour(name, age, job, birthday);

	std::cout << "name = " << name << std::endl;
	std::cout << "age = " << age << std::endl;
	std::cout << "job = " << job << std::endl;
	std::cout << "birthday = " << birthday << std::endl;

	p.reset({"location", "birth city"});
	p.parse("input.txt");
	p.pour(location, birth_city);

	std::cout << "location = " << location << std::endl;
	std::cout << "birth city = " << birth_city << std::endl;

	return 0;
}

