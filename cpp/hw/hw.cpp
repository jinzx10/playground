#include <iostream>

struct tq
{
	tq(): i(0) {}
	tq(int j): i(j) {}

	int i;
	tq operator-(int j) {
		return tq(i-j);
	}
};

template <int N>
int factorial() {
    return N*factorial<N-1>();
}

template <>
int factorial<1>() {
  return 1;
}


int main() {
	std::cout << factorial<1>() << std::endl;
	return 0;
}
