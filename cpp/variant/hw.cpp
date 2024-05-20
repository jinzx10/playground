#include <iostream>
#include <type_traits>
#include <variant>
#include <unordered_map>
#include <cassert>
#include <vector>

template <typename ...Ts>
class Variant {

private:

    template <typename T, typename R> 
    static constexpr bool type_match() { 
        return std::is_same<T, R>::value;
    }

    template <typename T, typename R1, typename R2, typename ...Rs> 
    static constexpr bool type_match() { 
        return std::is_same<T, R1>::value || type_match<T, R2, Rs...>(); 
    }

    struct Base {
        virtual ~Base() {};
        virtual Base* clone() const = 0;
    };

    template <typename T>
    struct Data : Base {
        Data(T const& t): val(t) {}
        Data* clone() const { return new Data(*this); }
        T val;
    };

    Base* ptr;

public:

    Variant(): ptr(nullptr) {}
    Variant(Variant<Ts...> const& var): ptr( var.ptr ? var.ptr->clone() : nullptr ) {}

    template <typename T, typename std::enable_if<type_match<T, Ts...>(), int>::type = 0 >
    Variant(T const& val): ptr(new Data<T>(val)) {}


    ~Variant() { delete ptr; }

    template <typename T, typename std::enable_if< type_match<T, Ts...>(), int>::type = 0 >
    Variant<Ts...>& operator=(T const& t) {
        delete ptr;
        ptr = new Data<T>(t);
        return *this;
    }

    Variant<Ts...>& operator=(Variant<Ts...> const& var) {
        delete ptr;
        ptr = (var.ptr) ? var.ptr->clone() : nullptr;
        return *this;
    }

    template <typename T, typename std::enable_if<type_match<T, Ts...>(), int>::type = 0 >
    T& get() const {
        assert(ptr && typeid(*ptr) == typeid(Data<T>));
        return static_cast<Data<T>*>(ptr)->val;
    }

    template <typename T>
    bool is_type() const {
        return ptr && typeid(*ptr) == typeid(Data<T>);
    }

};

int main() {



    using VARIANT = Variant<int, double, std::string>;

    VARIANT v1(300);

    std::cout << v1.get<int>() << std::endl;
    std::cout << v1.is_type<int>() << std::endl;
    std::cout << v1.is_type<float>() << std::endl;

    VARIANT v2 = "good day";


    return 0;
}
