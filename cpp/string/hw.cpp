#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>

using namespace std;

int main() {

    std::vector<std::string> str = {"good", "day", "123", "hello"};
    char delim = '|';

    std::string tmp = std::accumulate(str.begin()+1, str.end(), str[0],
                    [delim](std::string &ss, const std::string &s) {
                        return ss + delim + s;
                    });

    std::cout << tmp << std::endl;



	return 0;
}
