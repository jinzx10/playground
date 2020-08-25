#include <iostream>
#include <unordered_map>


union Base
{
    bool b;
    int i;
    std::string s;
    double d;

};


using namespace std;

int main() {

    std::unordered_map<std::string, Base> ub;
    ub["is_restricted"] = true;

    return 0;
}
