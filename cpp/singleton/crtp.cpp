#include <iostream>

template <typename T>
class Singleton {

public:
    Singleton(const Singleton&) = delete;
    Singleton(Singleton&&) = delete;

    Singleton& operator=(const Singleton&) = delete;
    Singleton& operator=(Singleton&&) = delete;

    static T& get_instance() {
        static T instance;
        return instance;
    }

    static T const& get_const_instance() {
        static T instance;
        return instance;
    }

protected:
    Singleton() = default;
};

class Test : public Singleton<Test> {

public:
    //Test() = default;

    // inheritance from Singleton does not guard against explicit copy constructor (or assignment)!
    // derived class would no longer be a singleton if copy constructor/assignment is implemented
    //Test(const Test&) {
    //    std::cout << "copy constructor" << std::endl;
    //}

    void addone() { ++num_; std::cout << num_ << std::endl; }

private:
    int num_ = 0;

};

int main() {

    Test::get_instance().addone();
    Test::get_instance().addone();
    Test::get_instance().addone();

    //Test::get_const_instance().addone();


    //Test& t = Test::get_instance();
    //t.addone();
    //t.addone();

    //Test t2(t);

    //t2.addone();

    return 0;
}
