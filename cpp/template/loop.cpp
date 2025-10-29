#include <iostream>

template <int ...Is>
struct Ints {};

template <int ...Is>
void prints() {
    std::cout << "prints(): ";
    ((std::cout << Is << "  "), ...);
    std::cout << std::endl;
}

template <int ...Is>
void loopn(Ints<Is...>, Ints<>, Ints<>) { prints<Is...>(); }

// n-loop
template <int I, int J, int ...Is, int ...Js, int ...Ks>
void loopn(
        Ints<Ks...> /* cumulated indices */,
        Ints<I,Is...> /* starting indices of remaining inner loops */,
        Ints<J,Js...> /* ending indices of remaining inner loops */
) {
    static_assert(sizeof...(Is) == sizeof...(Js));
    loopn(Ints<Ks..., I>(), Ints<Is...>{}, Ints<Js...>{});
    if constexpr(I < J) {
        loopn(Ints<Ks...>(), Ints<I+1,Is...>{}, Ints<J,Js...>{});
    }
    //loopn(Ints<Ks..., I>(), Ints<Is...>{}, Ints<Js...>{});
}

int main() {
    std::cout << "loop from 2 to 5" << std::endl;
    loopn(Ints<>(), Ints<2>(), Ints<5>());

    std::cout << "double loop over (0,2)x(3,5)" << std::endl;
    loopn(Ints<>(), Ints<0,3>(), Ints<2,5>());

    std::cout << "triple loop over (0,2)x(3,5)x(8,10)" << std::endl;
    loopn(Ints<>(), Ints<0,3,8>(), Ints<2,5,10>());
}
