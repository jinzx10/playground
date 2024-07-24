#define BUFSIZE 1020
#define TABLESIZE BUFSIZE
#undef BUFSIZE
#define BUFSIZE 37

#include <iostream>

int main() {

    std::cout << "TABLESIZE: " << TABLESIZE << std::endl;

#undef BUFSIZE
#define BUFSIZE 42

    std::cout << "TABLESIZE: " << TABLESIZE << std::endl;
}
