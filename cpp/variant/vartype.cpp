#include <type_traits>
#include <iostream>
#include <armadillo>
#include <cassert>
//#include <variant>

template <typename ...Ts>
class VarType;

template <typename ...Ts>
std::ostream& operator<< (std::ostream& os, VarType<Ts...> const& var) {
    var.template printer<Ts...>(os);
    return os;
}
    
template <typename ...Ts>
class VarType 
{
    friend std::ostream& operator<< <>(std::ostream& os, VarType const& );

private:

    template <typename T, typename R>
    static constexpr bool exact_match() {
        return std::is_same<T, R>::value;
    }

    template <typename T, typename R1, typename R2, typename ...Rs>
    static constexpr bool exact_match() {
        return std::is_same<T, R1>::value || exact_match<T, R2, Rs...>();
    }

    struct Base {
        virtual ~Base() {};
        virtual Base* clone() const = 0;
    };

    template <typename T>
    struct Data : Base {
        Data(T const& t): val(t) {}
        T val;
        Data* clone() const { return new Data(*this); }
    };

    template <typename T, typename R, typename ...Rs>
    typename std::enable_if< std::is_assignable<R&,T>::value
        && !( std::is_same<R,bool>::value && std::is_array<T>::value ), 
    int >::type updater(T const& t) {
        return (typeid(*ptr) == typeid(Data<R>)) ? (get<R>() = t, 0) : updater<T, Rs...>(t);
    }

    template <typename T, typename R, typename ...Rs>
    typename std::enable_if< !std::is_assignable<R&,T>::value 
        || ( std::is_same<R,bool>::value && std::is_array<T>::value ),
    int >::type updater(T const& t) {
        return updater<T, Rs...>(t);
    }

    template <typename T>
    int updater(T const& t) { std::cout << "invalid update: " << t << std::endl; return 1; }

    template <typename R, typename R1, typename ...Rs>
    std::ostream& printer(std::ostream& os) const {
        return (typeid(*ptr) == typeid(Data<R>)) ? 
            ( std::is_standard_layout<R>::value ? os << get<R>() : os << '\n' << get<R>() ) : 
            printer<R1, Rs...>(os);
    }

    template <typename R>
    std::ostream& printer(std::ostream& os) const {
        return (typeid(*ptr) == typeid(Data<R>)) ? 
            ( std::is_standard_layout<R>::value ? os << get<R>() : os << '\n' << get<R>() ):
            os;
    }

    Base* ptr;


public:

    VarType(): ptr(nullptr) {}
    VarType(VarType<Ts...> const& var): ptr( (var.ptr) ? var.ptr->clone() : nullptr ) {}

    template <typename T, typename std::enable_if<exact_match<T, Ts...>(), int>::type = 0>
    VarType(T const& val): ptr(new Data<T>(val)) {}

    ~VarType() { delete ptr; }

    template <typename T, typename std::enable_if<exact_match<T, Ts...>(), int>::type = 0>
    VarType<Ts...>& operator=(T const& t) {
        delete ptr;
        ptr = new Data<T>(t);
        return *this;
    }

    VarType<Ts...>& operator=(VarType<Ts...> const& var) {
        delete ptr;
        ptr = (var.ptr) ? var.ptr->clone() : nullptr;
        return *this;
    }

    template <typename T, typename std::enable_if<exact_match<T, Ts...>(), int>::type = 0>
    T& get() const {
        assert(ptr && typeid(*ptr) == typeid(Data<T>));
        return static_cast<Data<T>*>(ptr)->val;
    }

    template <typename T>
    int update(T const& val) { return updater<T, Ts...>(val); }

    void print() const { std::cout << (*this) << std::endl; }
};


int main() {

    int* ptr = nullptr;
    std::cout << typeid(*ptr).name() << std::endl;

    VarType<bool, int, size_t, double, std::string, arma::mat> vd;

    {
    VarType<bool, int, size_t, double, std::string, arma::mat> va(std::string("good"));

    std::cout << "string: " << va.get<std::string>() << std::endl;

    va.get<std::string>() = "good!";
    std::cout << "string: " << va.get<std::string>() << std::endl;
    vd = va;
    }
    std::cout << "string: " << vd.get<std::string>() << std::endl;

    VarType<bool, int, size_t, double, std::string, arma::mat> vb = size_t(5);
    std::cout << "size_t: " << vb.get<size_t>() << std::endl;
    vb.get<size_t>() = 12;
    std::cout << "size_t: " << vb.get<size_t>() << std::endl;

    VarType<bool, int, size_t, double, std::string, arma::mat> vc(vb);
    std::cout << "size_t: " << vc.get<size_t>() << std::endl;




    return 0;
}


