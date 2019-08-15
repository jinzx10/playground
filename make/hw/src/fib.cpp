#include <fib.h>

unsigned int fib(unsigned int n) {
    if (n == 0)
	return 0;
    if (n == 1 || n == 2)
	return 1;
    else
	return fib(n-1) + fib(n-2);
}

