#include <ios>
#include <type_traits>
#include <iostream>

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
        return (typeid(*ptr) == typeid(Data<R>)) ? static_cast<Data<R>*>(ptr)->val = t, 0 :  updater<T, Rs...>(t);
    }

    template <typename T, typename R, typename ...Rs>
    typename std::enable_if<std::is_assignable<R&,T>::value, int>::type updater(T&& t) {
        return (typeid(*ptr) == typeid(Data<R>)) ? static_cast<Data<R>*>(ptr)->val = std::move(t), 0 :  updater<T, Rs...>(t);
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
    int update(T const& t) {
        return updater<T, Ts...>(t);
    }

};

int main() {

    std::cout << "int, int: " << std::is_assignable<std::add_lvalue_reference<int>::type, int>::value << std::endl;
    std::cout << "string, const char[]: " << std::is_assignable<std::add_lvalue_reference<std::string>::type, const char[5]>::value << std::endl;
    std::cout << "bool, const char[]: " << std::is_assignable<bool, const char[5]>::value << std::endl;
    std::cout << "bool&, const char[]: " << std::is_assignable<bool&, const char[5]>::value << std::endl;


    /*
    VarType<int, std::string> var(std::string("good"));

    auto ca = "yet good";
    std::cout << var.update("bad") << std::endl;
    std::cout << var.update(ca) << std::endl;
    std::cout << ca << std::endl;
    */


    return 0;
}
