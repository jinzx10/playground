#include <any>
#include <armadillo>
#include <unordered_map>
#include <variant>

using namespace arma;
using namespace std;

int main() {

    unordered_map<string, std::any> anymap;
    unordered_map<string, std::variant<arma::mat, bool, int, double> > varmap;
    unordered_map<string, std::variant<arma::mat, bool, int, double, arma::vec> > varmap2;

    anymap["good"] = true;

    cout << anymap["good"].type().name() << endl;

    varmap["mat"] = randu(3,3);
    cout << std::get<mat>(varmap["mat"]) << endl;
    //varmap2["mat"] = randu(3,3);

    anymap["mat"] = randu(3,3);
    cout << anymap["mat"].type().name() << endl;
    any_cast<mat>(anymap["mat"]).print();


    return 0;
}
