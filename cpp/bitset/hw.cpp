#include <bitset>
#include <string>
#include <iostream>

using namespace std;

int main() {

    int i = 255;
    string str = bitset<8>(i).to_string();

    cout << str << endl;

}
