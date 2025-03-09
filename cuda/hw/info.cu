#include <stdio.h>

int main() {
	int i = 0;
	cudaGetDeviceCount(&i);
	printf("device number = %i\n", i);
	return 0;
}
