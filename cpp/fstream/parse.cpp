#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <tuple>

struct P
{
	P(std::vector<std::string> const& keys_) : keys(keys_), sz(keys.size()), vals(sz) {}
	std::vector<std::string> keys;
	size_t sz;
	std::vector<std::string> vals;
	void parse(std::string const& file);
};

void P::parse(std::string const& file) {
	std::fstream fs(file);
	std::string current_line;

	while (std::getline(fs, current_line)) {
		size_t i;
		for (i = 0; i != keys.size(); ++i) {
			auto pos = current_line.find(keys[i]);
			if (pos != std::string::npos) {
				current_line.erase(pos, keys[i].length());
				vals[i] = current_line;
				break;
			}
		}
	}
}

template <int N = 0, typename T>
void pour(std::vector<std::string> const& vals, T& val) {
	std::stringstream ss;
	ss << vals[N];
	ss >> val;
}

template <int N = 0, typename T, typename ...Ts>
void pour(std::vector<std::string> const& vals, T& val, Ts& ...args) {
	std::stringstream ss;
	ss << vals[N];
	ss >> val;
	pour<N+1, Ts...>(vals, args...);
}

template <typename T, int N = 0>
void pour2(std::vector<std::string> const& vals, T& val) {
	std::stringstream ss;
	ss << vals[N];
	ss >> val;
}

template <typename T, typename ...Ts, int N = 0>
void pour2(std::vector<std::string> const& vals, T& val, Ts& ...args) {
	std::stringstream ss;
	ss << vals[N];
	ss >> val;
	pour2<Ts..., N+1>(vals, args...);
}



int main() {

	P p({"size", "value", "dir"});
	p.parse("input.txt");

	int sz;
	double val;
	std::string dir;
	double tmp;

	//pour(p.vals, sz, val, dir);
	pour2(p.vals, sz, val, tmp);

	std::cout << "size = " << sz << std::endl;
	std::cout << "value = " << val << std::endl;
	std::cout << "dir = " << dir << std::endl;

	return 0;
}
