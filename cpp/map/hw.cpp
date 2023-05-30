#include <tuple>
#include <map>
#include <iostream>

//#include <unordered_map> // tuple is not hashable by default

int main() {

    std::map<std::tuple<int, int>, int> imap;


    imap.insert(std::make_pair(std::make_tuple(1, 2), 2));
    imap.insert(std::make_pair(std::make_tuple(2, 3), 4));
    imap.insert(std::make_pair(std::make_tuple(5, 2), 6));

    auto it = imap.find(std::make_tuple(1, 2));
    std::cout << typeid(it).name() << std::endl;
    std::cout << it->second << std::endl;


    std::map<std::tuple<int, int, int, int, int, int, int>, int> index_map;

    return 0;
}
