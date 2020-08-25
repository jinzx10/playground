#include <iostream>
#include <armadillo>
#include <type_traits>
#include <variant>
#include <unordered_map>
#include <cassert>

using namespace arma;

template <typename ...Ts>
struct vararg
{
    template <typename T, typename T1> 
    static constexpr bool type_match() { 
        return std::is_same<T, T1>::value; 
    }

    template <typename T, typename T1, typename T2, typename ...Rs> 
    static constexpr bool type_match() { 
        return std::is_same<T, T1>::value || type_match<T, T2, Rs...>(); 
    }

    struct Base { virtual ~Base() {}; };

    template <typename T>
    struct Data : Base
    {
        Data(T const& t): val(t) {}
        T val;
    };

    vararg(): ptr(nullptr) {}

    template <typename T, typename std::enable_if< type_match<T, Ts...>(), int>::type = 0 >
    vararg(T const& val): ptr(new Data<T>(val)) {}

    vararg(vararg<Ts...> const& var): ptr(nullptr) { clone<Ts...>(var); }

    ~vararg() { delete ptr; }

    template <typename T, typename std::enable_if< type_match<T, Ts...>(), int>::type = 0 >
    vararg<Ts...>& operator=(T const& t) {
        delete ptr;
        ptr = new Data<T>(t);
        return *this;
    }

    vararg<Ts...>& operator=(vararg<Ts...> const& var) {
        clone<Ts...>(var);
        return *this;
    }

    template <typename T, typename std::enable_if< type_match<T, Ts...>(), int>::type = 0 >
    T& get() const {
        assert(ptr && typeid(*ptr) == typeid(Data<T>));
        return static_cast<Data<T>*>(ptr)->val;
    }

    private:
    Base* ptr;

    // make a local copy of the Data stored in v
    // use RTTI to find the active type of v
    template <typename R, typename R1, typename ...Rs>
    void clone(vararg<Ts...> const& v) {
        if (!v.ptr) {
            delete ptr;
            ptr = nullptr;
            return;
        }

        if ( typeid(*v.ptr) == typeid(Data<R>) )
            clone<R>(v);
        else
            clone<R1, Rs...>(v);
    }

    template <typename R>
    void clone(vararg<Ts...> const& v) {
        if (!v.ptr) {
            delete ptr;
            ptr = nullptr;
            return;
        }

        delete ptr;
        ptr = new Data<R>(v.get<R>());
    }

};


int main() {

    vararg<int, double, arma::mat> va(eye(3,3).eval());
    va.get<mat>().print();
    va.get<mat>().zeros();
    va.get<mat>().print();

    va = 5;
    std::cout << "now va = " << va.get<int>() << std::endl;
    //std::cout << va.get<mat>() << std::endl; // assertion fail!
    

    vararg<int, double, arma::mat> vb(va);
    std::cout << "now vb = " << vb.get<int>() << std::endl;
    vb = 3.14;
    std::cout << "now vb = " << vb.get<double>() << std::endl;;

    vararg<int, double, arma::mat> vc;

    vc = vb;
    std::cout << "vc: \n";
    std::cout << "now vc = " << vc.get<double>() << std::endl;
    vc = randu(3,3).eval();
    vc.get<mat>().print();

    std::unordered_map<std::string, vararg<int, double, mat, bool> > varmap;

    varmap["pi"] = 3.14;
    varmap["isgood"] = true;
    varmap["eye3"] = eye(3,3).eval();

    std::unordered_map<std::string, vararg<int, double, mat, bool> > varmap2(varmap);

    std::cout << varmap2["pi"].get<double>() << std::endl;
    std::cout << varmap2["eye3"].get<mat>() << std::endl;

    

    /*
    using VarArg = std::variant<int, arma::vec, arma::mat, double, bool, char, size_t, arma::uvec>;
    VarArg vo = 3.14;
    std::cout << std::get<double>(vo) << std::endl; // must be int!
    vo = 'c';
    std::cout << std::get<char>(vo) << std::endl; // must be int!
    */

    return 0;
}
