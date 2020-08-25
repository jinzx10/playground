#include <iostream>
#include <type_traits>
#include <functional>
#include <cmath>
#include <string>

template <typename C>
struct Test
{
    template <typename T, typename std::enable_if<std::is_convertible<T, std::string>::value, int>::type = 0>
    typename std::enable_if<std::is_convertible<T,C>::value, void>::type test(T const& t) {
        std::cout << typeid(T).name() << std::endl;
        str = t;
    }
    C str;
};

int main() {

    Test<std::string> t;
    t.test("bad");

    std::cout << t.str << std::endl;

	return 0;
}
