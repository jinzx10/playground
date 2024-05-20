#include <iostream>
#include <utility>

template <size_t ...Is>
void prints() {
    std::cout << "prints(): ";
    ((std::cout << Is << "  "), ...);
    std::cout << std::endl;
}

template <size_t ...Is>
struct Ints {};

template <typename T1, typename T2>
struct Concat {};

template <size_t ...Is, size_t ...Js>
struct Concat<Ints<Is...>, Ints<Js...>> { using type = Ints<Is..., Js...>; };

template <size_t ...Is>
void recur(const Ints<Is...>&) { prints<Is...>(); }

template <size_t...Js, typename...Ts>
void recur(const Ints<Js...>& , Ints<>, Ts...) { return; }

template <size_t I, size_t ...Is, size_t...Js, typename...Ts>
void recur(const Ints<Js...>& tup, Ints<I, Is...>, Ts...args) {
    recur(typename Concat<Ints<Js...>, Ints<I>>::type{}, args...);
    recur(tup, Ints<Is...>{}, args...);
}

int main() {
    recur(Ints<>{}, Ints<0,1,2>{});
    recur(Ints<>{}, Ints<0,1,2>{}, Ints<10,11,12>{});
    recur(Ints<>{}, Ints<0,1,2>{}, Ints<10,11,12>{}, Ints<20,21,22>{});
    recur(Ints<>{}, Ints<0,1>{}, Ints<2>{}, Ints<3,4,5>{}, Ints<6,7>{});

    //recur(
    //    Ints<>{},
    //    Ints<0,1,2,3,4,5,6,7,8,9>{},
    //    Ints<0,1,2,3,4,5,6,7,8,9>{},
    //    Ints<0,1,2,3,4,5,6,7,8,9>{},
    //    Ints<0,1,2,3,4,5,6,7,8,9>{}
    //);
}

