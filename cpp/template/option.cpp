#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include "armadillo"
#include <cassert>

template <typename ...Ts>
class VarType {

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
    typename std::enable_if<std::is_convertible<T, R>::value &&
    ( (!std::is_same<R, bool>::value && !std::is_same<T, bool>::value) || (std::is_same<T,bool>::value && std::is_same<R,bool>::value) ), int>::type updater(T const& t) {
        return (typeid(*ptr) == typeid(Data<R>)) ? (get<R>() = t, 0) : updater<T, Rs...>(t);
    }

    template <typename T, typename R, typename ...Rs>
    typename std::enable_if<!std::is_convertible<T, R>::value || 
    ( (!std::is_same<R, bool>::value && std::is_same<T, bool>::value) || (!std::is_same<T,bool>::value && std::is_same<R,bool>::value) ), int>::type updater(T const& t) {
        return updater<T, Rs...>(t);
    }
    template <typename T>
    int updater(T const&) { return 1; }

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
};

using VarOpt = VarType< bool, std::string >;

using Input = std::unordered_map<std::string, VarOpt>;


class Themes_option
{
    public:
    Themes_option();
    template <typename ...Ts> Themes_option(Ts const& ...args);

    template <typename T>
    int set(std::string const& key, T const& val);

    template <typename T, typename ...Ts>
    int set(std::string const& key, T const& val, Ts const& ...args);

    Input const& get() const { return opts; }

    void print();

    protected:

    Input opts;

    int set() { return 0; }
};

Themes_option::Themes_option():
    opts{
        {"str", std::string("good")},
        {"good", true}
    }
{}

template <typename ...Ts>
Themes_option::Themes_option(Ts const& ...args): Themes_option() {
    set(args...);
}

template <typename T>
int Themes_option::set(std::string const& key, T const& val) {
    return (opts.find(key) == opts.end()) ?  1 : opts[key].update(val);
}

template <typename T, typename ...Ts>
int Themes_option::set(std::string const& key, T const& val, Ts const& ...args) {
    return set(key, val) + set(args...);
}


int main() {
    Themes_option opts(
            "str", "good"
    );
    

    return 0;
}
