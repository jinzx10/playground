#include <iostream>

using namespace std;

int* getptr(int sz) {
	int* ptr = new int[sz];
	for (int i = 0; i != sz; ++i) ptr[i] = 42;
	return ptr;
}

void newptr(int*& ptr, int sz) {
	delete[] ptr;
	ptr = new int[sz];
	for (int i = 0; i != sz; ++i) ptr[i] = 40;
}

void newptr2(int* ptr, int sz) {
	delete[] ptr;
	ptr = new int[sz];
	for (int i = 0; i != sz; ++i) ptr[i] = 35;
}

void print(int* ptr, int sz) {
	for (int i = 0; i != sz; ++i)
		cout << ptr[i] << " ";
	cout << endl;
}

int main() {
	int sz = 6;
	int* ptri = &sz;
	cout << "address of a variable on stack: " << ptri << endl;
	//delete ptri; // this would give a compile error; ptri is not dynamically allocated and delete is invalid.

	ptri = getptr(sz);

	cout << "address of returned pointer: " << ptri << endl;

	cout << "current values: ";
	print(ptri, sz);

	newptr(ptri, sz);
	cout << "address of allocated memory: " << ptri << endl;
	cout << "current values: ";
	print(ptri, sz);

	int* ptri2 = ptri;
	delete[] ptri2; // behavior of ptri is now undefined

	cout << "current values of old pointer: ";
	print(ptri, sz);

	return 0;
}
