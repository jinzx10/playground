#include "hw.h"
#include <iostream>

template <typename T>
void print() {
    std::cout << "General template" << std::endl;
}


template <>
void print<double>() {
    std::cout << "Specialized template for double" << std::endl;
}


// instantiation for int, char
template void print<int>();
template void print<char>();
