#include <iterator>
#include <cstdio>

int main() {

    int arr[] = {1, 2, 3, 4, 5};
    std::reverse_iterator<int*> rbegin(arr + 5);
    std::reverse_iterator<int*> rend(arr);

    for (std::reverse_iterator<int*> it = rbegin; it != rend; ++it) {
        printf("%d\n", *it);
    }
    printf("\n");

    for (std::reverse_iterator<int*> it = rbegin; it != rend; ++it) {
        printf("%d\n", *it.base());
    }
}
