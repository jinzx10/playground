#include <iterator>
#include <iostream>
#include <armadillo>
#include <algorithm>

using d2 = double[2];
using i3 = int[3];

auto f = [](i3 ai) {return ai[0]+ai[1]+ai[2];};

double sum(d2 const& da) {
	return da[0] + da[1];
}

int main() {
	i3 a = {1,2,3};
	d2 d = {1.1, 2.2};

	for (auto& c : a) {
		auto i = &c - std::begin(a);
		std::cout << "a[" << i << "] = " << c << std::endl;
	}

	decltype(a) b;

	std::copy(std::begin(a), std::end(a), std::begin(b));
	for (auto& c : b) {
		auto i = &c - std::begin(b);
		std::cout << "b[" << i << "] = " << c << std::endl;
	}

	std::cout << f(a) << std::endl;
	std::cout << sum(d) << std::endl;

	//std::vector<int> vi;
	//std::copy(std::begin(a), std::end(a), std::begin(vi));
	//for (auto& c : vi) {
	//	auto i = &c - &*std::begin(vi);
	//	std::cout << "vi[" << i << "] = " << c << std::endl;
	//}

	//arma::vec va;
	//std::copy(std::begin(a), std::end(a), std::begin(va));
	//for (auto& c : va) {
	//	auto i = &c - &*std::begin(va);
	//	std::cout << "va[" << i << "] = " << c << std::endl;
	//}



	return 0;
}


