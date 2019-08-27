#include <iomanip>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

int main() {
	stringstream ss;
	fstream file("mat.txt");
	string str;
	double num;
	vector<double> mat, row;

	int sz_row = 0, sz_col = 0, sz_col_now = 0;

	while (getline(file, str)) {
		sz_row += 1;
		std::cout << str << std::endl;
		ss << str;
		while (ss >> num)
			row.push_back(num);
		sz_col_now = row.size();
		if (sz_row == 1) sz_col = sz_col_now;
		if (sz_col != sz_col_now) std::cerr << "inconsistent matrix row size" << std::endl;
		mat.insert(mat.end(), row.begin(), row.end());
		row.clear();
		ss.str("");
		ss.clear();
	}

	for (int r = 0; r != sz_row; ++r) {
		for (int c = 0; c != sz_col; ++c) {
			std::cout << std::setw(4) << mat[r*sz_col+c] << " ";
		}
		std::cout << std::endl;
	}

	return 0;
}
