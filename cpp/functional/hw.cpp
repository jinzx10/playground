#include <functional>
#include <iostream>

using namespace std;

class Test {
public:
	Test(int i) : i_(i) {}

	int i_;

    void addone() { i_++; }
    static void minusone(int i) { std::cout << --i << std::endl; }

    void op() {
        std::function<void()> f = std::bind(&Test::addone, this);
        std::function<void(Test*)> f2 = &Test::addone;
        std::function<void(int)> g = &Test::minusone;
};

void func( int(*)(int) ) {
}

int main() {

	Test t(f1, f2);

	cout << t.F(4) << endl;
	

	return 0;
}
