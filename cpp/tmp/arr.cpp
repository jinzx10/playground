#include <iostream>

int main() {
	size_t const sz_col = 3;
	size_t const sz_row = 5;
	int (*A)[sz_col] = nullptr;

	A = new int[sz_row][sz_col];

	for (size_t i = 0; i != sz_row; ++i)
		for (size_t j = 0; j != sz_col; ++j)
			A[i][j] = i*10+j;


	for (size_t i = 0; i != sz_row; ++i) {
		for (size_t j = 0; j != sz_col; ++j) {
			std::cout << A[i][j] << " ";
		}
		std::cout << std::endl;
	}

	int* B = new int;
	*B = 3;
	delete[] B;
	return 0;
}
