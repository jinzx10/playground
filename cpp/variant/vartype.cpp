#include <type_traits>
#include <iostream>
#include <armadillo>
#include <cassert>
#include <variant>


template <typename ...Ts>                                                                        
struct VarType                                                                                   
{                                                                                                
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
        virtual Base* copy() const = 0;
    };
                                                                                                 
    template <typename T> 
    struct Data : Base
    {
        Data(T const& t): val(t) {}
        T val;
        Data<T>* copy() const { return new Data<T>(*this); }
    };  

    VarType(): ptr(nullptr) {}                                                                   
    //VarType(VarType<Ts...> const& var): ptr(nullptr) { clone<Ts...>(var); }
    VarType(VarType<Ts...> const& var): ptr(var.ptr->copy()) {}

    template <typename T, typename std::enable_if< exact_type_match<T, Ts...>(), int>::type = 0 >
    VarType(T const& val): ptr(new Data<T>(val)) {}

    ~VarType() { delete ptr; }

    template <typename T, typename std::enable_if< exact_type_match<T, Ts...>(), int>::type = 0 >
    VarType<Ts...>& operator=(T const& t) {
        delete ptr;
        ptr = new Data<T>(t);
        return *this;
    }

    VarType<Ts...>& operator=(VarType<Ts...> const& var) {
        //clone<Ts...>(var);
        delete ptr;
        ptr = var.ptr->copy();
        return *this;
    }

    template <typename T, typename std::enable_if< exact_type_match<T, Ts...>(), int>::type = 0 >
    T& get() const {
        assert(ptr && typeid(*ptr) == typeid(Data<T>));
        return static_cast<Data<T>*>(ptr)->val;
    }

    private:
    Base* ptr;

    // use RTTI to find the active type of v
    template <typename R, typename R1, typename ...Rs>
    void clone(VarType<Ts...> const& v) {
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
    void clone(VarType<Ts...> const& v) {
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


