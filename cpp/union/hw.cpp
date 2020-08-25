#include <iostream>
#include <armadillo>
#include <unordered_map>
#include <utility>
#include <variant>
#include <any>

using namespace std;
using namespace arma;

class HopInfo
{
    friend class Lattice;

    public:
    HopInfo();
    HopInfo(double ampl_, size_t from_, size_t to_, arma::uvec const& R_);

    void print();

    private:
    double ampl;
    size_t orb_from;
    size_t orb_to;
    arma::uvec R;
};

HopInfo::HopInfo(double a, size_t f, size_t t, uvec const& R):
    ampl(a), orb_from(f), orb_to(t), R(R)
{}

void HopInfo::print() {
    cout << ampl << endl;
    cout << orb_from << endl;
    cout << orb_to << endl;
    cout << R << endl;
}

union varopts
{
	int i;
	double d;
	std::string s;
	char c;
	bool b;
	arma::vec v;
	arma::uvec uv;
	arma::mat m;
	size_t sz;
    std::vector<HopInfo> vhi;

	varopts(bool val) : b(val) {}
    varopts& operator=(bool x) { b = x; return *this; }
	operator bool() { return b; }

	varopts(int val) : i(val) {}
    varopts& operator=(int x) { i = x; return *this; }
	operator int() { return i; }
	
	varopts(double val) : d(val) {}
    varopts& operator=(double x) { d = x; return *this; }
	operator double() { return d; }

	varopts(size_t val) : sz(val) {}
    varopts& operator=(size_t x) { sz = x; return *this; }
	operator size_t() { return sz; }

	varopts(char val) : c(val) {}
    varopts& operator=(char x) { c = x; return *this; }
	operator char() { return c; }

	varopts(std::string const& val) : s(val) {}
    varopts& operator=(string const& x) { s = x; return *this; }
	operator std::string() { return s; }

    // string literal
	varopts(char const* val) : s(val) {}
    varopts& operator=(char const* x) { s = x; return *this; }

	varopts(arma::vec const& val) : v(val) {}
	operator arma::vec() { return v; }
    varopts& operator=(vec const& x) { v = x; return *this; }

	varopts(arma::uvec const& val) : uv(val) {}
	operator arma::uvec() { return uv; }
    varopts& operator=(uvec const& x) { uv = x; return *this; }

	varopts(arma::mat const& val) : m(val) {}
    varopts& operator=(mat const& x) { m = x; return *this; }
	operator arma::mat() { return m; }

	varopts(std::vector<HopInfo> const& val) : vhi(val) {}
	operator std::vector<HopInfo>() { return vhi; }
    varopts& operator=(std::vector<HopInfo> const& x) { vhi = x; return *this; }

	varopts() {}
	~varopts() {}
};

//void assign(mat& m, const mat& mm) {
//    m = mm;
//}

//template <typename T>
//void assign(T& var, std::unordered_map<std::string, varopts> const& opt, std::string const& key) {
//    auto f = [&var] (T const& val) { var = val; };
//    f(opt.at(key));
//}

using Input = std::unordered_map<std::string, varopts>;

struct Test
{

    mat m;
    Test(Input const& opt = {}): m( (opt.find("mat")!= opt.end()) ? opt.at("mat").m : eye(3,3) ) {}
};

int main() {

    std::unordered_map<std::string, varopts> optmap;


    optmap.emplace(std::make_pair("is_restricted", true));
    bool a = optmap["is_restricted"];
    cout << boolalpha << a << endl;

    optmap["good"] = true;
    bool b = optmap["good"];
    cout << b << endl;

    optmap["mat"] = mat{eye(10, 10)};
    //mat m = optmap["mat"];
    mat m; 
    //m = optmap["mat"];
    //m = optmap["mat"].m;
    //assign(m, optmap, "mat");
    //assign(m, optmap.at("mat"));
    std::cout << "sum of diag = " << accu(m.diag()) << endl;

    optmap["info"] = "broyden";
    string info = optmap["info"];
    cout << "info: " << info << endl;

    HopInfo hi({0.1, 0,0,{1,2,3}});
    optmap["hopinfo"] = {hi};

    std::vector<HopInfo> vhi = optmap["hopinfo"];
    vhi[0].print();

    Test t(optmap);
    t.m.print();
    
    Test t2;
    t2.m.print();

	return 0;
}
