#include <iostream>
#include <ostream>
#include <type_traits>
#include <armadillo>
#include <cassert>

template <typename ...Ts>
class VarType;

template <typename ...Ts>
std::ostream& operator<< (std::ostream& os, VarType<Ts...> const& var) {
    var.template printer<Ts...>(os);
    return os;
}

template <typename ...Ts>
class VarType {   

    friend std::ostream& operator<< <>(std::ostream& os, VarType const& );

private:

    template <typename T, typename R>
    static constexpr bool exact_type_match() {
        return std::is_same<T, R>::value;
    }   
                                                                                                 
    template <typename T, typename R1, typename R2, typename ...Rs>
    static constexpr bool exact_type_match() {
        return std::is_same<T, R1>::value || exact_type_match<T, R2, Rs...>();
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
    typename std::enable_if<std::is_convertible<T, R>::value, int>::type updater(T const& t) {
        return (typeid(*ptr) == typeid(Data<R>)) ? get<R>() = t, 0 : updater<T, Rs...>(t);
    }

    template <typename T, typename R, typename ...Rs>
    typename std::enable_if<!std::is_convertible<T, R>::value, int>::type updater(T const& t) {
        return updater<T, Rs...>(t);
    }

    template <typename T>
    int updater(T const&) { return 1; }

    template <typename R, typename R1, typename ...Rs>
    std::ostream& printer(std::ostream& os) const {
        return (typeid(*ptr) == typeid(Data<R>)) ? os << get<R>() : printer<R1, Rs...>(os);
    }

    template <typename R>
    std::ostream& printer(std::ostream& os) const {
        return (typeid(*ptr) == typeid(Data<R>)) ? os << get<R>() : os;
    }

    Base* ptr;


public:

    VarType(): ptr(nullptr) {}
    VarType(VarType<Ts...> const& var): ptr( (var.ptr) ? var.ptr->clone() : nullptr ) {}
                                                                                                 
    template < typename T, typename std::enable_if< exact_type_match<T, Ts...>(), int >::type = 0 > 
    VarType(T const& val): ptr(new Data<T>(val)) {}
        
    ~VarType() { delete ptr; }
        
    template < typename T, typename std::enable_if< exact_type_match<T, Ts...>(), int >::type = 0 > 
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
         
    template < typename T, typename std::enable_if< exact_type_match<T, Ts...>(), int >::type = 0 > 
    T& get() const {
        assert(ptr && typeid(*ptr) == typeid(Data<T>));
        return static_cast<Data<T>*>(ptr)->val;
    }

    template <typename T>
    int update(T const& val) { return updater<T, Ts...>(val); }

    void print() const { std::cout << (*this) << std::endl; }
};

int main() {
    VarType<bool, int, double, size_t, std::string, arma::mat> var;
    var = std::string("good");

    std::cout << var.get<std::string>() << std::endl;

    var.update("bad");
    std::cout << var.get<std::string>() << std::endl;
    var.print();

    var = false;
    std::cout << var.get<bool>() << std::endl;
    std::cout << "update: " << var.update(true) << std::endl;;
    std::cout << var.get<bool>() << std::endl;

    var = size_t(5);
    std::cout << "update: " << var.update("bad") << std::endl;;
    std::cout << var.get<size_t>() << std::endl;


    VarType<bool, int, double, size_t, std::string, arma::mat> var2(var);
    std::cout << var2.get<size_t>() << std::endl;

    VarType<bool, int, double, size_t, std::string, arma::mat> var3;
    var3 = var2;
    std::cout << var3.get<size_t>() << std::endl;

    //var3.print(std::cout);

    std::cout << var3 << std::endl;

    var3 = arma::eye(3,3).eval();
    std::cout << var3 << std::endl;

    return 0;
}




