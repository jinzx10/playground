#include <iostream>
#include "./foo.h"

class Bar {
friend class Foo;

public:
    void print() {
        std::cout << "hello world" << std::endl;
    }
};

int main() {

    Bar b;
    b.print();

    std::cout << "good day" << std::endl;
    return 0;

}
