#include <iostream>
#include <type_traits>
#include <variant>
#include <unordered_map>
#include <cassert>
#include <vector>


class NodeType {
public:
    NodeType(int i) : i(i) {}

    NodeType(const std::string& s) : key(s) {}

    template <size_t N>
    NodeType(const char (&s)[N]) : key(s) {}

private:
    int i;
    std::string key;
};

class NodeUnion {
public:
    NodeUnion(int i) : node(i) {}

    NodeUnion(const std::string& s) : node(s) {}

    template <size_t N>
    NodeUnion(const char (&s)[N]) : node(s) {}

    // move constructor
    NodeUnion(NodeUnion&& other) : node(std::move(other.node)) {}

private:
    union Node {
        Node() {}
        Node(int i) : i(i) {}
        Node(const std::string& s) : s(s) {}
        Node(Node&& other) : s(std::move(other.s)) {}
        ~Node() {}

        int i;
        std::string s;
    } node;
};

template <typename T>
void add_json(const std::vector<NodeType>& nodes, const T& value) {
    std::cout << "add_json" << std::endl;
}


int main() {

    std::vector<NodeType> nodes{1, 3, "good"};
    add_json({"good", 1}, 3);

    std::vector<NodeUnion> nu{1, 3, "good"};





}
