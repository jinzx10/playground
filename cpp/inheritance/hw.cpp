#include <iostream>

using namespace std;

class Base {
public:
    Base(): tag('b') {}

    virtual Base* gen() {
        return new Base;    
    }

    virtual void overlap(Base* ket) {
        cout << "Base::overlap(Base*)" << endl;
    }

    char tag;
};

class Derived : public Base {
public:
    Derived() { tag = 'd'; } 
    Derived* gen() {
        return new Derived;
    }

    void overlap(Derived* ket) {
        cout << "Derived::overlap(Derived*)" << endl;
    }
};


void print(Base* b) {
    cout << b->gen()->tag << endl;
    b->overlap(nullptr);
}

int main() {

    Derived d0;
    print(&d0);


	return 0;
}
