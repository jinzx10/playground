#include <regex>
#include <iostream>

std::string extract(std::string const& str, std::string const& keyword) {
    std::smatch match;
    std::string regex_string = ".*" + keyword + "=\" *([^=]+) *\".*";
    //std::cout << regex_string << std::endl;

    std::regex re(regex_string);
    std::regex_match(str, match, re);

    //std::cout << match.size() << std::endl;
    //std::cout << match[0].str() << std::endl;
    //std::cout << match[1].str() << std::endl;

    for (auto it = match.begin(); it != match.end(); ++it)
        std::cout << "i = " << (it - match.begin()) << "      " << it->str() << std::endl;

    return match[1].str();
}

int main() {


    std::string str0 = "size=\"602\"";
    std::string str1 = "size=\"   602\"";
    std::string str2 = "size=\"   602   \"";
    std::string str3 = "          size=\"   602   \"";
    std::string str4 = "          size=\"   602   \"   l=\" 1 \"";

    std::cout << "extracted: " << extract(str0, "size") << std::endl;
    std::cout << "extracted: " << extract(str1, "size") << std::endl;
    std::cout << "extracted: " << extract(str2, "size") << std::endl;
    std::cout << "extracted: " << extract(str3, "size") << std::endl;
    std::cout << "extracted: " << extract(str4, "size") << std::endl;



    return 0;
}
