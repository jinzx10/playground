#include <array>
#include <iostream>
#include <cstdio>
#include <vector>


int main() {

    std::vector<int> v{1,2,3};

    std::cout << "before" << std::endl;
    for (auto i : v) {
        std::cout << i << std::endl;
    }

    std::vector<int> v2;

    bool b = true;
    std::vector<int>& x = b ? v : v2;

    v = std::move(x);
    std::cout << "after" << std::endl;
    for (auto i : v) {
        std::cout << i << std::endl;
    }

    return 0;
}
