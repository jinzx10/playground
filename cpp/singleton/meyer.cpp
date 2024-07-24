#include <thread>
#include <iostream>

#include <functional>
#include <vector>

// Meyer's singleton, "magic statics"
class Singleton {

public:
    Singleton(Singleton const&) = delete;
    void operator=(Singleton const&) = delete;

    static Singleton& GetInstance(std::string const& str)
    {
        static Singleton instance(str);
        return instance;
    }

    std::string value() const{ return value_; }

private:
    Singleton() {}
    Singleton(std::string const& str) : value_(str) {}

    std::string value_;
};

int main() {

    int nt = 10;
    std::vector<std::thread> t(nt);
    std::vector<std::function<void()>> f(nt);

    auto f_base = [](int i) { 
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        Singleton& singleton = Singleton::GetInstance(std::to_string(i));
        std::cout << singleton.value() << std::endl;
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


