#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

union varopts
{
	int i;
	double d;
	std::string str;
	char c;
	bool b;
	arma::vec v;
	arma::mat m;
	size_t sz;

	varopts(int const& val) : i(val) {}
	operator int() { return i; }
	
	varopts(double const& val) : d(val) {}
	operator double() { return d; }

	varopts(std::string const& val) : str(val) {}
	operator std::string() { return str; }

	varopts(char const& val) : c(val) {}
	operator char() { return c; }

	varopts(bool const& val) : b(val) {}
	operator bool() { return b; }

	varopts(arma::vec const& val) : v(val) {}
	operator arma::vec() { return v; }

	varopts(arma::mat const& val) : m(val) {}
	operator arma::mat() { return m; }

	varopts(size_t const& val) : sz(val) {}
	operator size_t() { return sz; }

	varopts() {}
	~varopts() {}
};

int main() {

	varopts vo(true);

	bool b;

	b = vo;

	cout << boolalpha << b << endl;


	return 0;
}
