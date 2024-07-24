#include <iostream>
#include <cmath>
#include <cassert>

int main() {

    for (int i = 0; i < 100000; ++i)
        assert(i == static_cast<int>(std::sqrt((double)i*i)));

    return 0;


}
