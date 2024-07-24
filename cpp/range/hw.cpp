#include <initializer_list>
#include <iostream>
#include <set>


int main() {

    int m = 1;

    std::initializer_list<int> range_m;
    if (m == 0) {
       range_m = {0};
    } else {
        range_m = {-m, m};
    }

    for (int n : range_m)
    {
        std::cout << n << std::endl;
    }

}
