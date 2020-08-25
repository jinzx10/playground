#include <functional>
#include <iostream>

using namespace std;
using d2d = std::function<double(double)>;

struct Test
{
	//Test(d2d F1, d2d F2) { F = [=] (double x) {return F1(x)-F2(x);}; } // OK
	Test(d2d F1, d2d F2) : F([=](double x){return F1(x)-F2(x);}) {} // OK

	d2d F;
};

void func( int(*)(int) ) {
}

int main() {

	d2d f1 = [](double x) { return x*x;};
	d2d f2 = [](double x) { return 2*x-1;};

	Test t(f1, f2);

	cout << t.F(4) << endl;
	

	return 0;
}
