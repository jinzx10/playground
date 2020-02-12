#include <iostream>

template <typename F, typename ...Args>
using return_t = decltype( std::declval<F>()( std::declval<Args>()... ) );

struct foo
{


int main() {



	return 0;
}
