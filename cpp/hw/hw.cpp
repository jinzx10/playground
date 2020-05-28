#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>
#include "../utility/widgets.h"
#include "../utility/arma_helper.h"
#include "../utility/math_helper.h" 

using namespace arma;
using namespace std;

template <typename T>
class OptBase
{
public:
	OptBase(T const& v): val(v) {}

	template <typename U = T>
	typename std::enable_if<!std::is_same<U,T>::value, void>::type assign_to(U&) { }

	template <typename U = T>
	typename std::enable_if<std::is_same<U,T>::value, void>::type assign_to(U& u) { u = val; }

private:
	T val;
};

int main(int, char**argv) {
	
	vec v = {1,2,3};
	OptBase<arma::vec> opt(v);

	vec u;
	mat z;

	opt.assign_to(u);
	opt.assign_to(z);

	u.print();

	z.print();



    return 0;
}
