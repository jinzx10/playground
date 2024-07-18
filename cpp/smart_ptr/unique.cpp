#include <memory>
#include <iostream>

class Obj
{
public:

    Obj(int n): n_(n), p_(new int[n]) {
        for (int i = 0; i < n; ++i) {
            p_[i] = i;
        }
    }
    ~Obj() { delete[] p_; }

    int n_ = 0;
    int* p_ = nullptr;

};

class Test
{
public:
    void build(int n) {
        obj_ = std::unique_ptr<Obj>(new Obj(n));
    }

    std::unique_ptr<Obj> obj_;
};


int main() {

    Test t;
    t.build(6);

    std::cout << t.obj_->p_[1] << std::endl;
    std::cout << t.obj_->n_ << std::endl;

    //std::unique_ptr<Obj> obj = std::move(t.obj_);
    //std::cout << t.obj_->p_[1] << std::endl;
    //std::cout << t.obj_->n_ << std::endl;
    
    Obj* q = t.obj_.release();

    std::cout << t.obj_->n_  << std::endl;


    return 0;
}
