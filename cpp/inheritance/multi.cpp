#include <iostream>

using namespace std;

struct Base1
{
	Base1(int j): i(j) {}

	int i;
};

struct Base2
{
	void print_i() { cout << get_i() << std::endl; }
	virtual int get_i() = 0;
};

struct Derived: public Base1, public Base2
{
	Derived(int j): Base1(j) {}
	int get_i() { return Base1::i; }
};

/*
struct Bottom
{
	Bottom(int j):i(j) {}
	
	void print() { std::cout << "bottom: " << get_i() << std::endl; }
	virtual int get_i() { return i;}
	int i;
};

struct Middle : public Bottom
{
	Middle(int j): Bottom(2*j), i(j) {}
	virtual int get_i() { return i;}
	void print() { std::cout << "middle: " << get_i() << std::endl; }
	int i;
};


struct Top: public Middle
{
	Top(int j): Middle(2*j), i(j) {}
	int i;
	virtual int get_i() { return i;}
	void print() { std::cout << "top: " << get_i() << std::endl; }
};

*/

int main() {

	Derived d(6);
	d.print_i();

	return 0;
}
