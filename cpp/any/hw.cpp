#include <string>
#include <iostream>
#include <any>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <cassert>

template <typename T>
class TD;

class Any {
public:
    Any() noexcept: ptr(nullptr) {}
    ~Any() { delete ptr; }

    Any& swap(Any& rhs) noexcept {
        std::swap(ptr, rhs.ptr);
        return *this;
    }

    // enable unqualified calls to swap
    friend void swap(Any& lhs, Any& rhs) noexcept {
        lhs.swap(rhs);
    }

    // copy constructor
    Any(Any const& rhs): ptr(rhs.ptr ? rhs.ptr->clone() : nullptr) {}

    // move constructor
    Any(Any&& rhs) noexcept: ptr(rhs.ptr) { rhs.ptr = nullptr; }

    // copy assignment
    Any& operator=(Any const& rhs) {
        Any(rhs).swap(*this);
        return *this;
    }

    // move assignment
    Any& operator=(Any&& rhs) noexcept {
        rhs.swap(*this);
        Any().swap(rhs);
        return *this;
    }

    // prevent Any(Any&) and Any(const Any&&) from matching
    template <typename T, typename = typename std::enable_if<!std::is_same<T, Any&>::value && !std::is_same<T, const Any>::value>::type>
    Any(T&& t): ptr(new Data<typename std::decay<T>::type>(std::forward<T>(t))) {}

    // no need to treat operator=(Any&) and operator=(const Any&&) differently; constructors will handle them.
    template <typename T>
    Any& operator=(T&& rhs) {
        Any(std::forward<T>(rhs)).swap(*this);
        return *this;
    }

    std::type_info const& type() const noexcept {
        return ptr ? ptr->type() : typeid(void);
    }

    bool has_value() const noexcept {
        return ptr;
    }

    void reset() noexcept {
        Any().swap(*this);
    }

    template <typename T>
    friend T* any_cast(Any* any) noexcept;

    template <typename T>
    friend T const* any_cast(Any const* any) noexcept;

    template <typename T>
    friend T any_cast(Any& any);

    template <typename T>
    friend T any_cast(Any const& any);

    template <typename T>
    friend T any_cast(Any&& any);

private:

    struct Base {
        virtual ~Base() {}; 
        virtual Base* clone() const = 0;
        virtual std::type_info const& type() const = 0;
    };

    template <typename T>
    struct Data : Base
    {
        Data(T const& t): val(t) {}
        ~Data() override {}

        Base* clone() const override {
            return new Data(val);
        }

        std::type_info const& type() const override {
            return typeid(T);
        }

        T val;
    };

    Base* ptr;

    class BadAnyCast : public std::bad_cast {
    public:
        virtual const char* what() const noexcept {
            return "Bad any cast";
        }
    };
};

template <typename T>
T* any_cast(Any* any) noexcept {
    return any && any->type() == typeid(T)
        ? &static_cast<Any::Data<typename std::remove_cv<T>::type>*>(any->ptr)->val
        : nullptr;
}

template <typename T>
T const* any_cast(Any const* any) noexcept {
    return any_cast<T>(const_cast<Any*>(any));
}

template <typename T>
T any_cast(Any& any) {
    using U = typename std::remove_reference<T>::type;
    auto* p = any_cast<U>(&any);
    if (!p) {
        throw Any::BadAnyCast{};
    }
    return static_cast<T>(*p);
}

template <typename T>
T any_cast(Any const& any) {
    using U = typename std::remove_reference<T>::type;
    return any_cast<U const&>(const_cast<Any&>(any));
}

template <typename T>
T any_cast(Any&& any) {
    using U = typename std::remove_reference<T>::type;
    return any_cast<U&&>(std::move(any));
}


int main() {
    
    //using ANY = std::any;
    using ANY = Any;

    //const int i0 = 5;
    //ANY a0(i0);
    //assert(a0.type() == typeid(int));

    //int i1 = 8;
    //ANY a1(i1);
    //assert(a1.type() == typeid(int));

    //const int& i2 = 5;
    //ANY a2(i2);
    //assert(a2.type() == typeid(int));

    //const int&& i3 = 5;
    //ANY a3(i3);
    //assert(a3.type() == typeid(int));

    //int&& i4 = 5;
    //ANY a4(i4);
    //assert(a4.type() == typeid(int));

    //ANY a5(5);
    //assert(a5.type() == typeid(int));

    //ANY b0 = a0;
    //assert(b0.type() == typeid(int));

    //const ANY b1 = 5;
    //ANY b2 = b1;
    //assert(b2.type() == typeid(int));

    //ANY b3 = std::move(b1);
    //assert(b3.type() == typeid(int));

    //ANY b4 = std::move(b2);
    //assert(b4.type() == typeid(int));

    //const std::string str0= "good day";
    //ANY s0 = str0;
    //assert(s0.type() == typeid(std::string));

    //const std::string& str1 = "good day";
    //ANY s1 = str1;
    //assert(s1.type() == typeid(std::string));

    //const std::string&& str2 = "good day";
    //ANY s2 = str2;
    //assert(s2.type() == typeid(std::string));

    //const std::string str3 = "good day";
    //ANY s3 = std::move(str3);
    //assert(s3.type() == typeid(std::string));

    ANY c0 = 'c';
    ANY c1;
    assert(c1.type() == typeid(void));

    c1.reset();
    c1 = c0;
    assert(c1.type() == typeid(char));

    c1.reset();
    c1 = std::move(c0);
    assert(c1.type() == typeid(char));

    const ANY cc = 'b';
    c1.reset();
    c1 = cc;
    assert(c1.type() == typeid(char));

    c1.reset();
    c1 = std::move(cc);
    assert(c1.type() == typeid(char));

    std::cout << any_cast<int>(c1) << std::endl;;

    //std::unordered_map<std::string, ANY> anymap;

    //anymap["good"] = true;
    //anymap["index"] = 5;
    //anymap["pi"] = 3.14;

    //std::cout << anymap["good"].type().name() << std::endl;
    //std::cout << anymap["index"].type().name() << std::endl;
    //std::cout << anymap["pi"].type().name() << std::endl;

    return 0;
}
