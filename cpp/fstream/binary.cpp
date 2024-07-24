#include <fstream>
#include <iostream>
#include <array>
#include <cmath>
#include <map>

int main() {

    int num = 10000;

    using KeyType = std::array<short, 6>;

    KeyType key;
    double val;
    std::map<KeyType, double> m;

    std::ofstream out("binary.dat", std::ios::binary);
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < 6; ++j) {
            key[j] = i + j;
        }
        val = std::sqrt(i);
        m[key] = val;

        out.write(reinterpret_cast<char*>(&key), sizeof(key));
        out.write(reinterpret_cast<char*>(&val), sizeof(val));
    }
    out.close();

    // read back
    KeyType key2;
    double val2;
    std::map<KeyType, double> m2;

    std::ifstream in("binary.dat", std::ios::binary);
    for (int i = 0; i < num; ++i) {
        in.read(reinterpret_cast<char*>(&key2), sizeof(key2));
        in.read(reinterpret_cast<char*>(&val2), sizeof(val2));
        m2[key2] = val2;
    }


    // check
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < 6; ++j) {
            key[j] = i + j;
        }
        val = std::sqrt(i);
        if (m[key] != m2[key]) {
            std::cout << "Error: " << i << std::endl;
        }
    }

    

    return 0;
}
