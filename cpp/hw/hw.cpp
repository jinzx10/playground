#include <chrono>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>
#include <cassert>
#include <armadillo>

using namespace std;
using namespace arma;

struct State
{
    template <typename T, typename = typename std::enable_if< std::is_integral<T>::value, T>::type >
    State(T const& state_) {
        if (state_ != 0 && state_ != 1) {
            std::cerr << "exceed:" << state_ << std::endl;
            exit(EXIT_FAILURE);
        }
        state = state_;
    }

    operator bool() const {
        return state;
    }

    bool state;
};

void foo(State const& S) {
    bool state = S;
    std::cout << state << std::endl;
}

int main(int, char**argv) {

    arma::wall_clock timer;
    timer.tic();
    sleep(1);
    double n = timer.toc();
    std::cout << "time elapsed = " << n << std::endl;

    timer.tic();
    sleep(3);
    n = timer.toc();
    std::cout << "time elapsed = " << n << std::endl;
}
