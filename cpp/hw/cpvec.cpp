#include <iostream>
#include <vector>

int main() {

    std::vector<int> v0(10, 1);
    v0.reserve(1000000);

    std::vector<int> v1(1000000, 2);

    std::cout << "&start   of v0: " << &v0[0] << std::endl;
    std::cout << "capacity of v0: " << v0.capacity() << std::endl;
    std::cout << "&start   of v1: " << &v1[0] << std::endl;
    std::cout << "capacity of v1: " << v1.capacity() << std::endl;

    v0 = v1;
    std::cout << std::endl;

    std::cout << "&start   of v0: " << &v0[0] << std::endl;
    std::cout << "capacity of v0: " << v0.capacity() << std::endl;
    std::cout << "&start   of v1: " << &v1[0] << std::endl;
    std::cout << "capacity of v1: " << v1.capacity() << std::endl;

    return 0;
}
