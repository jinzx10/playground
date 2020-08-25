#include <utility>
#include <iostream>

using namespace std;

template <typename T>
void test(T&& t) {
    cout << typeid(t).name() << ": " << t << endl;
}

int main() {

    int i = 5;
    test(i);

    return 0;
}
