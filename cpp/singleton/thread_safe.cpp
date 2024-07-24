#include <thread>
#include <iostream>
#include <mutex>
#include <functional>
#include <vector>

class Singleton {

private:
    static Singleton * instance_;
    static std::mutex mutex_;

protected:
    Singleton(const std::string value): value_(value) {}
    ~Singleton() {}
    std::string value_;

public:
    Singleton(Singleton &other) = delete;
    void operator=(const Singleton &) = delete;

    static Singleton *GetInstance(const std::string& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        //mutex_.lock();
        if (instance_ == nullptr)
        {
            instance_ = new Singleton(value);
        }
        //mutex_.unlock();
        return instance_;
    }
    
    std::string value() const{ return value_; } 
};

Singleton* Singleton::instance_{nullptr};
std::mutex Singleton::mutex_;

int main()
{   
    int nt = 10;
    std::vector<std::thread> t(nt);
    std::vector<std::function<void()>> f(nt);

    auto f_base = [](int i) { 
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        Singleton* singleton = Singleton::GetInstance(std::to_string(i));
        std::cout << singleton->value() << std::endl;
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
