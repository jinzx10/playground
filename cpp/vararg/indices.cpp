#include <armadillo>

using namespace arma;

template <typename T>
uvec indices(T const& i) {
	return uvec{i};
}

template <typename T, typename ...Ts>
uvec indices(T const& i, Ts const& ...args) {
	return join_cols(uvec{i}, indices(args...));
}

int main() {
	uword a = 5;
	uvec idx = indices(3u, a, 8u, uvec{11u,14u});
	idx.print();

	auto range = [](uword const& i, uword const& j) {
		return regspace<uvec>(i,j);
	};

	uvec idx2 = indices(a, range(2,4), uvec{3,5}, 0);
	idx2.print();
	

	return 0;
}
