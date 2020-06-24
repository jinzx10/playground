#include <iostream>

using namespace std;

struct Base
{
	int times2() { return 2*get_num(); }
	virtual int get_num() = 0;
};

struct Base2
{
	int get_num() { return 10;}
};

struct Derived : public Base
{

};

int main() {


	Derived d;
	cout << d.get_num() << endl; // 5
	cout << d.times2() << endl; // 10

}
