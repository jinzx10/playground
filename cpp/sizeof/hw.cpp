#include <vector>
#include <iostream>

int main() {

    std::vector<double> vd;
    std::vector<int> vi(10);

    std::vector<int[2]> v2i(10);
    std::vector<int[2]> v2i2;

    v2i2 = v2i;

    //std::cout << sizeof(vi) << std::endl;
    //std::cout << vi.size() << std::endl;;
    //std::cout << vi.capacity() << std::endl;;

    //vi.push_back(0);
    //std::cout << vi.size() << std::endl;;
    //std::cout << vi.capacity() << std::endl;;

    //double* ptrd = new double[100];

    //std::cout << sizeof(vd) << std::endl;
    //std::cout << sizeof(vi) << std::endl;
    //std::cout << sizeof(vi.begin()) << std::endl;
    //std::cout << sizeof(vi.size()) << std::endl;

    //vi.resize(10);
}
