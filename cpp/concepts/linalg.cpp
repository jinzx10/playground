#include <concepts> // c++20
#include <type_traits> // c++11
#include <complex>
#include <iostream>
#include <cblas.h>

extern "C" {
void sscal_(const int *N, const float *alpha, float *X, const int *incX);
void dscal_(const int *N, const double *alpha, double *X, const int *incX);
void cscal_(const int *N, const std::complex<float> *alpha, std::complex<float> *X, const int *incX);
void zscal_(const int *N, const std::complex<double> *alpha, std::complex<double> *X, const int *incX);
}

template <typename T>
concept BlasRealType = std::is_same<T, float>::value || std::is_same<T, double>::value;

template <typename T>
concept BlasComplexType = std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value;

template <typename T>
concept BlasType = BlasRealType<T> || BlasComplexType<T>;

template <BlasType T>
void scal(int sz, T fac, T* x) {
    int incx = 1;
    if (std::is_same<T, float>::value) { return sscal_(&sz, (float*) &fac, (float*)x, &incx); }
    if (std::is_same<T, double>::value) { return dscal_(&sz, (double*) &fac, (double*)x, &incx); }
    if (std::is_same<T, std::complex<float>>::value) { return cscal_(&sz, (std::complex<float>*) &fac, (std::complex<float>*)x, &incx); }
    if (std::is_same<T, std::complex<double>>::value) { return zscal_(&sz, (std::complex<double>*) &fac, (std::complex<double>*)x, &incx); }
}

template <typename T>
struct is_blas_type { 
    static constexpr bool value =
        std::is_same<T, float>::value ||
        std::is_same<T,double>::value || 
        std::is_same<T, std::complex<float> >::value ||
        std::is_same<T, std::complex<double> >::value;
};


template <typename T>
typename std::enable_if<is_blas_type<T>::value, void>::type scal2(int sz, T fac, T* x) {
    int incx = 1;
    if (std::is_same<T, float>::value) { return sscal_(&sz, (float*) &fac, (float*)x, &incx); }
    if (std::is_same<T, double>::value) { return dscal_(&sz, (double*) &fac, (double*)x, &incx); }
    if (std::is_same<T, std::complex<float>>::value) { return cscal_(&sz, (std::complex<float>*) &fac, (std::complex<float>*)x, &incx); }
    if (std::is_same<T, std::complex<double>>::value) { return zscal_(&sz, (std::complex<double>*) &fac, (std::complex<double>*)x, &incx); }
}



int main() {

    float s[] = {-1.1, 2.2, -3.3, 4.4, -5.5};
    double d[] = {-1.1, 2.2, -3.3, 4.4, -5.5};
    std::complex<float> c[] = {{-1.1,0}, {2.2,0}, {-3.3,0}, {4.4,0}, {-5.5,0}};
    std::complex<double> z[] = {{-1.1,0}, {2.2,0}, {-3.3,0}, {4.4,0}, {-5.5,0}};

    int u[] = {1,2,3,4,5};

    scal(5, 2.0f, s);
    scal(5, 2.0, d);
    scal(5, std::complex<float>{2.0,0}, c);
    scal(5, std::complex<double>{2.0,0}, z);

    //scal(5, 2, u);

    for (int i = 0; i != 5; ++i) {
        std::cout << s[i] << "   " << d[i] << "   " << c[i] << "   " << z[i] << std::endl;
    }

    scal2(5, 2.0f, s);
    scal2(5, 2.0, d);
    scal2(5, std::complex<float>{2.0,0}, c);
    scal2(5, std::complex<double>{2.0,0}, z);

    //scal2(5, 2, u);

    for (int i = 0; i != 5; ++i) {
        std::cout << s[i] << "   " << d[i] << "   " << c[i] << "   " << z[i] << std::endl;
    }

    return 0;
}
