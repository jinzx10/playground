#ifndef __AUXILLARY_H__
#define __AUXILLARY_H__

#include <type_traits>
#include <iterator>

template <typename ...>
using void_t = void;

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, void_t<decltype(
		std::begin(std::declval<T&>()) != std::end(std::declval<T&>()), // begin(), end(), !=
		++std::declval<decltype(std::begin(std::declval<T&>()))&>(), // ++
		//++std::begin(std::declval<T&>()), // this would fail
		*std::begin(std::declval<T&>()) // dereference (*)
		//*std::declval<std::begin(std::declval<T&>())>() // this would fail
	)>> : std::true_type {};


#endif
