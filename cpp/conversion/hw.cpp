#include <iostream>

class Bar {
public:
    Bar(int i) : sz_(i), data_(new int[i]) {}

    int sz_;
    int* data_;
};


class Foo {
public:
    Foo(int i) : sz_(i), data_(new int[i]) {}

    Foo(Bar const& bar) : sz_(bar.sz_), data_(bar.data_) {}

    int sz_;
    int* data_;
};

