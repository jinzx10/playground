#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

struct P
{
	P(std::vector<std::string> const& keys_) : keys(keys_), n_keys(keys.size()), vals(n_keys) {}
	std::vector<std::string> keys;
	size_t n_keys;
	std::vector<std::string> vals;
	void parse(std::string const& file) {
		std::fstream fs(file);
		std::string str;
		while (std::getline(fs, str)) {
			for (size_t i = 0; i != keys.size(); ++i) {
				auto pos = str.find(keys[i]);
				if (pos != std::string::npos) {
					str.erase(pos, keys[i].length());
					vals[i] = str;
					break;
				}
			}
		}
	}

	template <int N = 0, typename T>
	void pour(T& val) {
		if (N >= n_keys) {
			std::cerr << "Too many variables." << std::endl;
			return;
		}
		std::stringstream ss;
		ss << vals[N];
		ss >> val;
	}

	template <int N = 0, typename T, typename ...Ts>
	void pour(T& val, Ts& ...args) {
		if (N >= n_keys) {
			std::cerr << "Too many variables." << std::endl;
			return;
		}
		std::stringstream ss;
		ss << vals[N];
		ss >> val;
		pour<N+1, Ts...>(args...);
	}
};


int main() {
	P p({"size", "value", "dir"});
	p.parse("input.txt");

	int sz;
	double val;
	std::string dir;
	double tmp;

	p.pour(sz, val, dir, tmp);

	std::cout << "size = " << sz << std::endl;
	std::cout << "value = " << val << std::endl;
	std::cout << "dir = " << dir << std::endl;

	return 0;
}

