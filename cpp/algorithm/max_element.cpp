#include <iostream>
#include <algorithm>


int main() {

    int i[3] = {1, 4, 3};

    int me = *std::max_element(i, i + 3);

    std::cout << me << std::endl;


}
