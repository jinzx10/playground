#include <algorithm>
#include <functional>
#include <iostream>

int main() {

    double arr[] = {0.0, 1.1, 2.2, 3.5};

    std::cout << *(std::upper_bound(arr, arr + 4, 1.1)-1) << std::endl;
    std::cout << *(std::upper_bound(arr, arr + 4, 1.2)-1) << std::endl;

    std::cout << arr[0] << std::endl;


    return 0;
}
