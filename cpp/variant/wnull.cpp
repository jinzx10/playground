#include <iostream>
#include <type_traits>

using namespace std;


template <typename T>
typename std::enable_if<!std::is_array<T>::value, bool>::type test(T const& t) {
//bool test(T const& t) {
    bool a;
    a = t;
    std::cout << typeid(T).name() << std::endl;
    return a;
}

template <typename T>
typename std::enable_if<std::is_array<T>::value, bool>::type test(T const&) {
    return true;
}

int main() {

    const char info[] = "good";
    std::cout << typeid(info).name() << std::endl;

    test(info);
    test("good");
    //test(nullptr);

    bool b = "good";
    std::cout << b << std::endl;

    int i3[3] = {1,2,3};

    test(i3);

    return 0;
}
