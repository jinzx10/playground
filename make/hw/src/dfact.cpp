#include <dfact.h>

unsigned int dfact(unsigned int n) {
    return (n == 0 || n == 1 ) ? 1 : n*dfact(n-2);
}
