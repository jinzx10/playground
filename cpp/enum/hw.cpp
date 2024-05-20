#include <iostream>

enum class Color {
    RED,
    GREEN,
    BLUE
};

int main() {

    Color color = Color::RED;

    std::cout << "Color: " << static_cast<int>(color) << std::endl;

    return 0;
}
