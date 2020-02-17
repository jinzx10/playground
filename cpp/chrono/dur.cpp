#include <chrono>
#include <iostream>
#include <cstdlib>

using iclock = std::chrono::high_resolution_clock;

int main() {

	std::chrono::duration<double,std::ratio<1,1>> dur11;
	std::chrono::duration<double,std::ratio<1,10>> dur110;
	std::chrono::duration<double,std::ratio<1,1000>> durmili;
	std::chrono::duration<double,std::ratio<10,1>> dur101;

	iclock::time_point start = iclock::now();

	std::system("sleep 0.66");

	dur11 = iclock::now() - start;
	dur110 = iclock::now() - start;
	dur101 = iclock::now() - start;
	durmili= iclock::now() - start;

	std::cout << dur11.count() << std::endl;
	std::cout << dur110.count() << std::endl;
	std::cout << dur101.count() << std::endl;
	std::cout << durmili.count() << std::endl;
	
	std::cout << typeid(std::deca).name() << std::endl;

	return 0;
}
