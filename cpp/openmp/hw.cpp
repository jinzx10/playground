#include <iostream>
#include <omp.h>

int get_num_threads() {
    return 2;
}

int main() {

    int max = 1e5;

    unsigned long long sum = 0;
    #pragma omp parallel num_threads(get_num_threads())
    {
        std::cout << omp_get_num_threads() << std::endl;
        #pragma omp for
        for (int i = 0; i < max; ++i) {
            #pragma omp atomic
            sum += i;
        }

    }

    std::cout << "sum = " << sum << std::endl;


    return 0;
}
