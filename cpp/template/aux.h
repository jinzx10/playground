#ifndef __AUXILLARY_H__
#define __AUXILLARY_H__

#include <type_traits>
#include <iterator>

template <typename ...>
using void_t = void;

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, void_t<decltype(std::begin(std::declval<T&>())), decltype(std::end(std::declval<T&>()))>> : std::true_type {};



#endif
