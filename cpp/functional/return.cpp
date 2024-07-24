#include <functional>
#include <iostream>

std::function<int(int)> gen_addn(int n)
{
    return [n](int x) { return x + n; };
}

int main()
{
    auto f = gen_addn(3);
    std::cout << f(15) << std::endl;

    return 0;
}
