#include <fact.h>

int fact(int n) {
	return (n>1) ? fact(n-1)*n : 1;
}
