#include <type_traits>
#include <armadillo>
#include <iostream>

using namespace std;

template <typename T, typename R = void, typename ...Rs>
struct Convertible
{
    using type = typename std::conditional< std::is_convertible<T, R>::value, R, typename Convertible<T, Rs...>::type >::type;
};

template <typename T>
struct Convertible<T>
{
    using type = void;
};


int main() {

    std::cout << typeid(Convertible<int, arma::vec, arma::mat, size_t>::type).name() << std::endl;
    std::cout << typeid(Convertible<int, int, arma::vec, arma::mat, size_t>::type).name() << std::endl;
    std::cout << typeid(Convertible<int, arma::vec, unsigned int, arma::mat, size_t>::type).name() << std::endl;
    std::cout << typeid(Convertible<int, arma::vec, arma::mat, std::string>::type).name() << std::endl;

    return 0;
}
