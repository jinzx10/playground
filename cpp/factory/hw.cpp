#include <iostream>
#include <vector>
#include <memory>

class RadialSet {
public:
    int i_;
    char c_;

protected:
    RadialSet() = default;

private:
};

class AtomicRadials : public RadialSet {
public:
    AtomicRadials(int i) {
        i_ = i;
        std::cout << "AtomicRadials constructor" << std::endl;
    }
};

class BetaRadials : public RadialSet {
public:
    BetaRadials(char c) {
        c_ = c;
        std::cout << "BetaRadials constructor" << std::endl;
    }
};

class RadialCollection {
public:
    std::vector<std::unique_ptr<RadialSet>> radsets_;
};



int main() {

    RadialCollection rc;

    rc.radsets_.emplace_back(new AtomicRadials(1));
    rc.radsets_.emplace_back(new BetaRadials('c'));

    std::cout << rc.radsets_[0]->i_ << std::endl;
    std::cout << rc.radsets_[0]->c_ << std::endl;
    std::cout << rc.radsets_[1]->i_ << std::endl;
    std::cout << rc.radsets_[1]->c_ << std::endl;


    return 0;
}
