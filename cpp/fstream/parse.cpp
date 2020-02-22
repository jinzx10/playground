#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <tuple>
#include <type_traits>
#include <typeinfo>

template <typename ...Ts>
struct Parser
{
	Parser(std::vector<std::string> const& keys_) : keys(keys_), vals(keys.size()) {
		if ( sizeof...(Ts) > keys.size() ) {
			std::cerr << "Too many types specified." << std::endl 
				<< "Program aborted." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	std::vector<std::string> keys;
	std::vector<std::string> vals;

	void parse(std::string const& file) {
		std::fstream fs(file);
		std::string line;
		std::string str;
		std::stringstream ss;
		while (std::getline(fs, line)) {
			for (size_t i = 0; i != keys.size(); ++i) {
				ss << line;
				while (std::getline(ss, str, ',')) {
					auto pos = str.find(keys[i]);
					if (pos != std::string::npos) {
						str.erase(pos, keys[i].length());
						vals[i] = str;
						break;
					}
				}
				ss.str("");
				ss.clear();
			}
		}
	}

	template <int N, typename T>
	typename std::enable_if< ( N >= sizeof...(Ts) ), void>::type type_check() {}

	template <int N, typename T>
	typename std::enable_if< ( N < sizeof...(Ts) ), void>::type type_check() {
		if ( !std::is_same<T, typename std::tuple_element<N, std::tuple<Ts...>>::type>::value ) {
			std::cerr << "Type mismatch: argument " << N
				<< ": expected type: " << typeid(typename std::tuple_element<N, std::tuple<Ts...>>::type).name() 
				<< "; detected type: " << typeid(T).name() << std::endl
				<< "Program aborted." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	template <int N>
	void size_check() {
		if ( N >= keys.size() ) {
			std::cerr << "Too many variables. Program aborted." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	template <int N, typename T>
	void check() {
		size_check<N>();
		type_check<N,T>();
	}

	template <int N = 0>
	void pour(std::string& val) {
		check<N, std::string>();
		std::stringstream ss;
		ss << vals[N];
		std::getline(ss, val);
	}

	template <int N = 0, typename T>
	void pour(T& val) {
		check<N, T>();
		std::stringstream ss;
		ss << vals[N];
		ss >> val;
	}

	template <int N = 0, typename T, typename ...Rs>
	void pour(T& val, Rs& ...args) {
		pour<N>(val);
		pour<N+1, Rs...>(args...);
	}
};

int main() {
	Parser<int, double, std::string, unsigned int, std::string, bool> p ({"size", "value", "dir", "age", "name", "job"});
	//Parser<> p ({"size", "value", "dir", "age", "name", "job"});
	p.parse("input.txt");

	int sz;
	double val;
	std::string dir;
	unsigned int age;
	std::string name;
	bool job;
	char tmp;

	//p.pour(sz, tmp, dir, age, name, job);
	p.pour(sz, val, dir, age, name, job);

	std::cout << "size = " << sz << std::endl;
	std::cout << "value = " << val << std::endl;
	std::cout << "dir = " << dir << std::endl;
	std::cout << "age = " << age << std::endl;
	std::cout << "name = " << name << std::endl;
	std::cout << std::boolalpha << "job = " << job << std::endl;

	return 0;
}

