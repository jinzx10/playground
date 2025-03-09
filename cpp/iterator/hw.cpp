#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <cstdio>

int main() {

    //std::vector<double> dist = {-4.4, 1.1, -0.1, -3.3, 6.6, -1.1, 5.5, -2.2};
    std::vector<double> dist = {4.4, 1.1, 0.1, -3.3, -6.6, -1.1, -5.5, -2.2};
    int idx[] = {6, 2, 0, 3, 4, 9, 5, 7};
    int sz = dist.size();

    printf("Before:\n");
    for (int i = 0; i < sz; ++i) {
        printf("idx[i] = %2i    dist[i] = %6.1f\n", idx[i], dist[i]);
    }
    printf("\n");

    int *head = idx;
    std::reverse_iterator<int*> tail(idx + sz), rend(idx);
    auto is_negative = [&dist, &idx](int& j) { return dist[&j - idx] < 0; };
    while ( ( head = std::find_if(head, idx + sz, is_negative) ) <
            ( tail = std::find_if_not(tail, rend, is_negative) ).base() ) {
        std::swap(*head, *tail);
        std::swap(dist[head - idx], dist[tail.base() - idx - 1]);
        ++head;
        ++tail;
        for (int i = 0; i < sz; ++i) {
            printf("idx[i] = %2i    dist[i] = %6.1f\n", idx[i], dist[i]);
        }
        printf("distance from start to head: %li\n", head - idx);
        printf("distance from start to tail: %li\n", tail.base() - idx - 1);
        printf("\n");
    }
    //int num = std::find_if(idx, idx + sz, is_negative) - idx;
    printf("distance from start to head: %li\n", head - idx);
    printf("distance from start to tail: %li\n", tail.base() - idx - 1);
    int num = std::find_if(tail.base(), idx + sz, is_negative) - idx;
    printf("num = %i\n", num);

    printf("After:\n");
    for (int i = 0; i < sz; ++i) {
        printf("idx[i] = %2i    dist[i] = %6.1f\n", idx[i], dist[i]);
    }
    printf("\n");

    return 0;
}
