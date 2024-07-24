#include <cstdio>
#include <string>
#include <iostream>

class Fmt {
public:
    template <typename ...Args>
    static std::string format(const std::string& fmt, Args&&... args) {
        static constexpr int buf_size = 1024;
        static char buf[buf_size];
        snprintf(buf, buf_size, fmt.c_str(), std::forward<Args>(args)...);
        return std::string(buf);
    }
};

int main() {

    std::string solver = "DA";

    for (int i = 1; i < 10; i++) {
        std::cout << Fmt::format("%s%-3i    energy = %12.9f    error = %12.5e\n",
                solver.c_str(), i, 1.0 + 1.0 / i, 1.0 / i);
    } 

    return 0;
}
