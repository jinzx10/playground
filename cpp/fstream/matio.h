#ifndef __MATRIX_IO_H__
#define __MATRIX_IO_H__

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

template <typename T> void read_mat(std::string const& filename, T*& A, int& sz_row, int& sz_col, bool column_major = true);
template <typename T> void print_mat(T* const& A, int const& sz_row, int const& sz_col, bool column_major = true, int const& width = 4);


template <typename T>
void read_mat(std::string const& filename, T*& A, int& sz_row, int& sz_col, bool column_major) {
	std::fstream file(filename);
	std::vector<T> mat, row;
	std::string str;
	std::stringstream ss;
	sz_row = 0;
	sz_col = 0;

	T elem;
	int sz_col_now = 0;

	while (std::getline(file, str)) {
		ss << str;
		while (ss >> elem) {
			row.push_back(elem);
		}
		sz_col_now = row.size();
		sz_row += 1;
		if (sz_row == 1) sz_col = sz_col_now;

		if (sz_col_now != sz_col) {
			std::cerr << "error: inconsistent row length" << std::endl;
			return;
		}

		mat.insert(mat.end(), row.begin(), row.end());
		row.clear();
		ss.str("");
		ss.clear();
	}

	delete[] A;
	A = nullptr;

	A = new T[sz_row*sz_col];
	for (int i = 0; i != sz_row*sz_col; ++i) A[i] = 0;

	if (column_major) {
		for (int r = 0; r != sz_row; ++r) {
			for (int c = 0; c != sz_col; ++c) {
				A[r+c*sz_row] = mat[r*sz_col+c];
			}
		}
	} else {
		std::copy(mat.begin(), mat.end(), A);
	}
}

template <typename T>
void print_mat(T* const& A, int const& sz_row, int const& sz_col, bool column_major, int const& width) {
	for (int r = 0; r != sz_row; ++r) {
		for (int c = 0; c != sz_col; ++c) {
			std::cout << std::setw(width) << (column_major ? A[r+c*sz_row] : A[r*sz_col+c] ) << " ";
		}
		std::cout << std::endl;
	}
}


#endif
