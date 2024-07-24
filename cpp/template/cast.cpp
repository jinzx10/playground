#include <iostream>
#include <complex>
#include <type_traits>

using std::complex;

template <typename T>
typename std::enable_if<std::is_same<T, float>::value, T>::type cast_to(std::complex<double> const& z)
{
    return static_cast<float>(z.real());
}

template <typename T>
typename std::enable_if<std::is_same<T, double>::value, T>::type cast_to(std::complex<double> const& z)
{
    return z.real();
}

template <typename T>
typename std::enable_if<std::is_same<T, std::complex<float>>::value, T>::type cast_to(std::complex<double> const& z)
{
    return std::complex<float>(z.real(), z.imag());
}

template <typename T>
typename std::enable_if<std::is_same<T, std::complex<double>>::value, T>::type cast_to(std::complex<double> const& z)
{
    return z;
}


template <typename T>
void cast(std::complex<double> const& from, T* to) {
    if (std::is_same<T, float>::value) { *(float*)to = static_cast<float>(from.real()); }
    if (std::is_same<T, double>::value) { *(double*)to = from.real(); }
    if (std::is_same<T, std::complex<float>>::value) { *(std::complex<float>*)to = std::complex<float>(from.real(), from.imag()); }
    if (std::is_same<T, std::complex<double>>::value) { *(std::complex<double>*)to = from; }
}

template <typename U, typename Device>
struct Psi
{
    void calc() {
        std::complex<double> z;
        U num = cast_to<U>(z);
    }
};

int main() {

    double pi = std::acos(-1);
    std::complex<double> cmplx_dble(pi, -pi);

    float s;
    double d;
    complex<float> c;
    complex<double> z;

    cast(cmplx_dble, &s);
    cast(cmplx_dble, &d);
    cast(cmplx_dble, &c);
    cast(cmplx_dble, &z);

    //s = cast_to<float>(cmplx_dble);
    //d = cast_to<double>(cmplx_dble);
    //c = cast_to<std::complex<float>>(cmplx_dble);
    //z = cast_to<std::complex<double>>(cmplx_dble);


    printf("s = %20.8f\n", s);
    printf("d = %20.15f\n", d);
    printf("c = %20.8f + %20.8f\n", c.real(), c.imag());
    printf("z = %20.15f + %20.15f i\n", z.real(), z.imag());

    return 0;
}

