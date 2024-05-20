#include <iostream>
#include <utility>
#include <tuple>

template <size_t ...Is>
void print(const std::tuple<std::index_sequence<Is>...>&) {
    ((std::cout << Is << "  "), ...);
    std::cout << std::endl;
}

int main() {
    print(std::make_tuple(std::index_sequence<3>{}, std::index_sequence<1>{}));
	return 0;
}
