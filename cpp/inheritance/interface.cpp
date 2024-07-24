#include <iostream>

class Base {
public:
    virtual void build(int, int) {
        std::cout << "Base::build" << std::endl;
    }
};

class Derived1 : public Base {

public:

    void build(int a, int b) {
        std::cout << "Derived1::build" << std::endl;
    }

};

class Derived2 : public Base {

public:

    void build(char a, double b) {
        std::cout << "Derived2::build" << std::endl;
    }

};

int main() {

    Base* b1 = new Derived1();
    Base* b2 = new Derived1();


    b1->build(1, 2);
    b2->build('a', 3.14);

    return 0;

}
