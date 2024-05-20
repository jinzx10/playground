#include <chrono>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>
#include <cassert>
#include <memory>


int main() {

    typedef double complex[2];

    double* a = new double[2];

    complex* b = reinterpret_cast<complex*>(a);

    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;

    return 0;
}
