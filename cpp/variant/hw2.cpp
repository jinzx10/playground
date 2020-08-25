#include <iostream>
#include <type_traits>

struct Base
{

    using type = int;
    void print() { std::cout << "base" << std::endl; }
};

struct Derived: Base
{
    using type = double;
    void print() { std::cout << "derived" << std::endl; }
};


int main() {

    Derived d;

    Base* pb1 = &d;

    Base* pb2 = pb1;

    static_cast<Derived*>(pb2)->print();
    
    std::cout << typeid(std::remove_pointer<decltype(pb2)>::type).name() << std::endl;

    return 0;
}
