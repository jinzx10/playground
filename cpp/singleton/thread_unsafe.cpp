#include <functional>
#include <thread>
#include <iostream>
#include <vector>
#include <string>

class Singleton
{
protected:
    Singleton(const std::string value): value_(value) { }

    static Singleton* singleton_;

    std::string value_;

public:

    Singleton(Singleton &other) = delete;
    void operator=(const Singleton &) = delete;

    static Singleton *GetInstance(const std::string& value) {
        if(singleton_== nullptr){
            singleton_ = new Singleton(value);
        }
        return singleton_;
    }

    std::string value() const{ return value_; }
};

Singleton* Singleton::singleton_= nullptr;

int main()
{
    int nt = 10;
    std::vector<std::thread> t(nt);
    std::vector<std::function<void()>> f(nt);

    auto f_base = [](int i) { 
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        Singleton* singleton = Singleton::GetInstance(std::to_string(i));
        printf("%s\n", singleton->value().c_str());
        //std::cout << singleton->value() << "\n";
    };
    for (int it = 0; it != nt; ++it) {
        f[it] = std::bind(f_base, it);
        t[it] = std::thread(f[it]);
    }

    for (int it = 0; it != nt; ++it) {
        t[it].join();
    }

    return 0;
}


