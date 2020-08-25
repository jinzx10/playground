#include <type_traits>
#include <iostream>
#include <utility>

template <typename ...Ts>
class VarType
{
    template <typename T, typename R>
    static constexpr bool exact_match() {
        return std::is_same<T, R>::value;
    }

    template <typename T, typename R1, typename R2, typename ...Rs>
    static constexpr bool exact_match() {
        return std::is_same<T, R1>::value || exact_match<T, R2, Rs...>();
    }

    struct Base
    {
        virtual ~Base() {}
    };

    template <typename T>
    struct Data: Base
    {
        Data(T const& v): val(v) {}
        T val;
    };


    template <typename T, typename R, typename ...Rs>
    typename std::enable_if<std::is_assignable<R&,T>::value, int>::type updater(T const& t) {
        return (typeid(*ptr) == typeid(Data<R>)) ? std::cout << "lref" << std::endl, static_cast<Data<R>*>(ptr)->val = t, 0 :  updater<T, Rs...>(t);
    }

    template <typename T, typename R, typename ...Rs>
    typename std::enable_if<std::is_assignable<R&,T>::value, int>::type updater(T&& t) {
        return (typeid(*ptr) == typeid(Data<R>)) ? std::cout << "rref" << std::endl, static_cast<Data<R>*>(ptr)->val = std::move(t), 0 :  updater<T, Rs...>(std::forward<T>(t));
    }

    template <typename T, typename R, typename ...Rs>
    typename std::enable_if<!std::is_assignable<R&,T>::value, int>::type updater(T&& t) {
        return updater<T, Rs...>(t);
    }

    template <typename T, typename R, typename ...Rs>
    typename std::enable_if<!std::is_assignable<R&,T>::value, int>::type updater(T const& t) {
        return updater<T, Rs...>(t);
    }

    template <typename T>
    int updater(T const& ) { return 1; }
    template <typename T>
    int updater(T&& ) { return 1; }

    Base* ptr;

public:
    template <typename T, typename std::enable_if<exact_match<T, Ts...>(), int>::type = 0>
    VarType(T const& val): ptr(new Data<T>(val)) {}

    template <typename T>
    int update(T&& t) {
        return updater<T, Ts...>(std::forward<T>(t));
    }

    template <typename T>
    int update(T const& t) {
        return updater<T, Ts...>(t);
    }

};

int main() {


    VarType<bool, std::string> var(std::string("good"));

    std::cout << var.update("bad") << std::endl;


    return 0;
}
