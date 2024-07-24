#include <memory>
#include <cstdio>

#define LEAK3
//#define SAFE3

struct B; // forward declaration
struct C; // forward declaration

#ifdef LEAK2
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
#endif

#ifdef SAFE2
struct A {
    ~A() { printf("A dtor\n"); }
    std::shared_ptr<B> ptr_;
};

struct B {
    ~B() { printf("B dtor\n"); }
    std::weak_ptr<A> ptr_;
};

int main() {

    std::shared_ptr<A> a = std::make_shared<A>();
    std::shared_ptr<B> b = std::make_shared<B>();

    a->ptr_ = b;
    b->ptr_ = a;

    return 0;
}
#endif

#ifdef LEAK3 
struct A {
    ~A() { printf("A dtor\n"); }
    std::shared_ptr<B> ptr_;
};

struct B {
    ~B() { printf("B dtor\n"); }
    std::shared_ptr<C> ptr_;
};

struct C {
    ~C() { printf("C dtor\n"); }
    std::shared_ptr<A> ptr_;
};

int main() {

    std::shared_ptr<A> a = std::make_shared<A>();
    std::shared_ptr<B> b = std::make_shared<B>();
    std::shared_ptr<C> c = std::make_shared<C>();

    a->ptr_ = b;
    b->ptr_ = c;
    c->ptr_ = a;

    return 0;
}
#endif

#ifdef SAFE3
struct A {
    ~A() { printf("A dtor\n"); }
    std::shared_ptr<B> ptr_;
};

struct B {
    ~B() { printf("B dtor\n"); }
    std::weak_ptr<C> ptr_;
};

struct C {
    ~C() { printf("C dtor\n"); }
    std::shared_ptr<A> ptr_;
};

int main() {

    std::shared_ptr<A> a = std::make_shared<A>();
    std::shared_ptr<B> b = std::make_shared<B>();
    std::shared_ptr<C> c = std::make_shared<C>();

    a->ptr_ = b;
    b->ptr_ = c;
    c->ptr_ = a;

    return 0;
}
#endif

