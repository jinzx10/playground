#include <fact.h>

unsigned int fact(unsigned int n) {
    return (n == 0) ? 1 : n*fact(n-1);
}
