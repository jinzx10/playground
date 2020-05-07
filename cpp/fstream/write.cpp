#include <fstream>
#include <iostream>

int main() {

	double omega = 0.1;
	double mass = 2.00;

	std::fstream fs;
	fs.open("write.txt");
	fs << "omega " << omega << std::endl
		<< "mass " << mass << std::endl;

	return 0;
}
