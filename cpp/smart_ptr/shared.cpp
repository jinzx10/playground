#include <memory>
#include <cstdio>
#include <iostream>


class Test {
public:
    ~Test() = default;

    static std::shared_ptr<Test> create() {
        return std::shared_ptr<Test>(new Test);
    }

    void print() {
        std::cout << "Test" << std::endl;
    }

private:
    Test() {}
};


int main() {

    std::shared_ptr<Test> test;
    std::cout << "test.use_count() = " << test.use_count() << std::endl;

    test  = Test::create();
    std::cout << "test.use_count() = " << test.use_count() << std::endl;

    test.get()->print();

    return 0;
}
