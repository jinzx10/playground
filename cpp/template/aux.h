#ifndef __AUXILLARY_H__
#define __AUXILLARY_H__

#include <type_traits>
#include <iterator>
#include <complex>

template <typename ...> using void_t = void;
template <bool is_cplx> using num_t = typename std::conditional<is_cplx, std::complex<double>, double>::type;

template <bool is_cplx> num_t<is_cplx> keep_cplx(std::complex<double> const& z) { return z; }
template <> num_t<false> keep_cplx<false>(std::complex<double> const& z) { return z.real(); }

template <typename T, typename = void> struct is_iterable : std::false_type {};
template <typename T> struct is_iterable<T, void_t<decltype(
		std::begin(std::declval<T&>()) != std::end(std::declval<T&>()), // begin(), end(), !=
		++std::declval<decltype(std::begin(std::declval<T&>()))&>(), // ++, ++std::begin(std::declval<T&>()) does not work!
		*std::begin(std::declval<T&>()) // dereference (*)
	)>> : std::true_type {}; // use T& instead of T to include built-in array as iterable


#endif
