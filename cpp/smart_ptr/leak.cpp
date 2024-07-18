#include <memory>
#include <cstdio>

struct B; // forward declaration
struct A {
    ~A() { printf("A dtor\n"); }
    std::shared_ptr<B> ptr_;
};

struct B {
    ~B() { printf("B dtor\n"); }
    std::shared_ptr<A> ptr_;
};

int main() {

    std::shared_ptr<A> a = std::make_shared<A>();
    std::shared_ptr<B> b = std::make_shared<B>();

    a->ptr_ = b;
    b->ptr_ = a;

    return 0;
}
