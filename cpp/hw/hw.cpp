#include <iostream>
#include <cstdio>
#include <vector>

class Test {

public:
    void print() { printf("hello world!\n"); }

};

#define private public


int main() {

    Test t;

    t.print();

    return 0;
}
