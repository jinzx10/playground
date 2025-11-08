#include <optional>
#include <string_view>
#include <toml++/toml.hpp>
#include <iostream>
#include <sstream>

using namespace std::literals;

int main() {

    static constexpr auto source = R"(
        str = "hello world"

        numbers = [ 1, 2, 3, "four", 5.0 ]
        vegetables = [ "tomato", "onion", "mushroom", "lettuce" ]
        minerals = [ "quartz", "iron", "copper", "diamond" ]

        [animals]
        cats = [ "tiger", "lion", "puma" ]
        birds = [ "macaw", "pigeon", "canary" ]
        fish = [ "salmon", "trout", "carp" ]

        [atoms]
            natom = 314
            charge = [1.0, 2.0]
            [atoms.mag]
            Fe = [0.0, -1.0, -2.0]
            Ca = [3.0,  6.0,  9.0]
    )"sv;


    toml::table tbl = toml::parse(source);
    //std::cout << tbl << "\n";


    std::optional<std::string> str = tbl["str"].value<std::string>();
    std::cout << *str << std::endl;

    // this doesn't work
    //std::optional<std::string> str2 = tbl["str2"].value<std::string>();
    //std::cout << *str2 << std::endl;

    std::string_view str3 = tbl["str"].value_or("str3!");
    std::string_view str4 = tbl["str4"].value_or("str4!");

    std::cout << "str3 = " << str3 << std::endl;
    std::cout << "str4 = " << str4 << std::endl;

    std::cout << tbl["atoms"]["mag"]["Fe"][0] << std::endl;

    std::cout << "str = " << tbl["str"] << std::endl;

    return 0;
}
