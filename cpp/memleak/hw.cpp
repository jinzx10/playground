#include <iostream>

class Base
{
public:
	Base() : n_(0), ptr(nullptr) {}
	Base(int n) : n_(n) { 
		ptr = new int[n_];
		for (int i = 0; i != n; ++i) {
			ptr[i] = i;
		}
	}

	~Base() { delete[] ptr; }

	int n_ = 0;
	int* ptr = nullptr;
	void print() {
		for (int i = 0; i != n_; ++i) {
			printf("%i ", ptr[i]);
		}
		printf("\n");
	}
};


class ManyBase
{
public:
	ManyBase() : n_(0), ptr(nullptr) {}
	ManyBase(int n): n_(n) { 
		//ptr = new Base[n_] {n_, n_, n_, n_};
		ptr = new Base[n_];
		for (int i = 0; i != n_; ++i) {
			new(&ptr[i]) Base(i);
		}
	}
	~ManyBase() { delete[] ptr;}
	int n_;
	Base* ptr;
};


int main() {

	int n = 5;

	ManyBase mb(n);

	for (int i = 0; i != n; ++i) {
		mb.ptr[i].print();
	}
	return 0;
}

