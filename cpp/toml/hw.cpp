#define TOML_EXCEPTIONS 0
#include <toml++/toml.hpp>
#include <iostream>

int main(int, char** argv) {

    // no-exception mode, the parse function returns a toml::parse_result
    toml::parse_result result = toml::parse_file("test.toml");
    if (!result) {
        std::cerr << "Parsing failed:\n" << result.error() << "\n";
        return 1;
    }
    std::cout << result.table() << std::endl;

    // with-exception mode
    // the parse function returns a toml::table
    //toml::table tbl;
    //try {
    //    tbl = toml::parse_file(argv[1]);
    //    std::cout << tbl << "\n";
    //} catch (const toml::parse_error& err) {
    //    std::cerr << "Parsing failed:\n" << err << "\n";
    //    return 1;
    //}

    return 0;
}

