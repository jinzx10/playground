#include <iostream>

using namespace std;

struct Base
{
    virtual ~Base() {}
};

struct Derived: Base
{

};

int main() {

    Derived d;
    Base b;

    Base* ptr_b = new Derived;

    std::cout << typeid(b).name() << std::endl;
    std::cout << typeid(d).name() << std::endl;
    std::cout << typeid(*ptr_b).name() << std::endl;
    std::cout << (typeid(*ptr_b) == typeid(Derived)) << std::endl;

    std::cout << typeid(decltype(*ptr_b)).name() << std::endl;

    return 0;
}

