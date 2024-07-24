#include <iostream>
#include <vector>

struct Test {

    std::vector<int> v{10, 5};

};

int main() {

    Test t1;

    Test t2;

    t1.v[0] = 10;

    t2 = std::move(t1);

    std::cout << t1.v.size() << std::endl;
	return 0;
}
