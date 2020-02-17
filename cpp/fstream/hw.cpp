#include "matio.h"

int main() {

	double* A = nullptr;
	int sz_row, sz_col;

	read_mat("mat.txt", A, sz_row, sz_col);
	//read_mat<double>("mat.txt", A, sz_row, sz_col);
	print_mat<double>(A, sz_row, sz_col);


	return 0;
}
