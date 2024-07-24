#include <iostream>

enum class DataType {
    INT = 0,
    FLOAT = 1,
};

template <DataType dt>
struct GetType { };

template <>
struct GetType<DataType::INT> {
    typedef int type;
};

template <>
struct GetType<DataType::FLOAT> {
    typedef float type;
};

class Tensor
{
  public:
    Tensor(DataType dt, int sz) {
        data_type_ = dt;    
        sz_ = sz;
    }

    template <typename T = void>
    T* data() {
        return static_cast<T*>(data_);
    }

  private:

    DataType data_type_;
    int sz_ = 0;
    void* data_ = nullptr;
};


int main() {

    GetType<DataType::INT>::type t = 0;

    constexpr DataType dt = DataType::FLOAT;

    std::cout << typeid(t).name() << std::endl;

    GetType<dt>::type t2;
    std::cout << typeid(t2).name() << std::endl;

    return 0;
}
