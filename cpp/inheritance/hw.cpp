#include <iostream>

using namespace std;

struct Base
{
    Base(int j): i(j) {}
    int i;
    int geti() { return i; }
};

struct Common
{
    Common(int i): j(i) {}

    int j;
    void printj() { cout << "j = " << j << endl; }

};

struct Derived : Base, Common
{
    Derived(int i): Base(i), Common(geti()) {}

};


int main() {

    Derived d(5);

    d.printj();


	return 0;
}
