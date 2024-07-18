#include <memory>
#include <iostream>

class Test {
public:
    Test();
    ~Test() = default;

    void print();

private:
    class TestImpl;
    std::shared_ptr<TestImpl> impl;
};

class Test::TestImpl {
public:
    //static std::shared_ptr<TestImpl> create() { return std::make_shared<TestImpl>(); }
    static std::shared_ptr<TestImpl> create() { return std::shared_ptr<TestImpl>(new TestImpl); }

    void print() { std::cout << "TestImpl" << std::endl; }

private:
    TestImpl() {};
};

//Test::Test(): impl(std::make_shared<TestImpl>()) {}
Test::Test(): impl(TestImpl::create()) {}

void Test::print() {
    impl->print();
}


int main() {

    Test t;
    t.print();

    return 0;
}
