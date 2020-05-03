#ifndef __WIDGETS_H__
#define __WIDGETS_H__

#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <tuple>
#include <map>
#include <cstring>

// read arguments from the command line
template <int N = 1>
void readargs(char** args, std::string& var) {
	std::stringstream ss;
	ss << args[N];
	std::getline(ss, var);
}

template <int N = 1, typename T>
void readargs(char** args, T& var) {
	std::stringstream ss;
	ss << args[N];
	ss >> var;
}

template <int N = 1, typename T, typename ...Ts>
void readargs(char** args, T& var, Ts& ...rest) {
	readargs<N>(args, var);
	readargs<N+1>(args, rest...);
}

// mkdir
inline int mkdir(std::string const& dir) {
	std::string command = "mkdir -p " + dir;
	return std::system(command.c_str());
}

// a stopwatch class
class Stopwatch
{
	using iclock		= std::chrono::high_resolution_clock;
	using timepoint_t	= iclock::time_point;
	using dur_t			= std::chrono::duration<double>;

	std::map<unsigned int, timepoint_t> t_start;
	std::map<unsigned int, dur_t> dur_store;
	std::map<unsigned int, bool> is_running;

	template <typename ...>
	using void_t	= void;

	template <typename F, typename ...Args>
	using return_t	= decltype( std::declval<F>()(std::declval<Args>()...) );

	template <typename F, typename ...Args>
	void try_it(unsigned int const& N, F f, Args const& ...args) {
		if (N != 0) {
			f(args...);
			try_it(N-1, f, args...);
		}
	}


	public:

	Stopwatch(): t_start(), dur_store(), is_running() {}

	void run(unsigned int const& i = 0, std::string const& info = "") {
		if (!info.empty())
			std::cout << info << std::endl;
		if (t_start.find(i) == t_start.end()) { // if stopwatch "i" does not exist
			t_start[i] = iclock::now();
			dur_store[i] = dur_t::zero();
			is_running[i] = true;
		} else { // if stopwatch "i" already exists
			if (is_running[i]) {
				std::cout << "Stopwatch #" << i << " is already running. Nothing to do." << std::endl;
			} else {
				t_start[i] = iclock::now();
				is_running[i] = true;
			}
		}
	}
	
	void run(std::string const& info) {
		run(0, info);
	}

	void pause(unsigned int const& i = 0, std::string const& info = "") { 
		if (!info.empty())
			std::cout << info << std::endl;
		if (t_start.find(i) == t_start.end()) {
			std::cout << "Stopwatch #" << i << " does not exist." << std::endl;
		} else {
			if (is_running[i]) {
				dur_store[i] += iclock::now() - t_start[i];
				is_running[i] = false;
			} else {
				std::cout << "Stopwatch #" << i << " is not running. Nothing to do." << std::endl;
			}
		}
	}

	void pause(std::string const& info) {
		pause(0, info);
	}

	void report(unsigned int const& i = 0, std::string const& info = "") { 
		if (!info.empty())
			std::cout << info << ": "; 
		if (t_start.find(i) == t_start.end()) {
			std::cout << "Stopwatch #" << i << " does not exist." << std::endl;
		} else {
			dur_t dur = is_running[i] ? 
				dur_store[i] + static_cast<dur_t>(iclock::now() - t_start[i]) : dur_store[i];
			std::cout << "elapsed time = " << dur.count() << " seconds (stopwatch #" << i << ")." << std::endl; 
		}
	}

	void report(std::string const& info) {
		report(0, info);
	}

	void reset(unsigned int const& i = 0, std::string const& info = "") { 
		if (!info.empty())
			std::cout << info << std::endl;
		if (t_start.find(i) == t_start.end()) {
			std::cout << "Stopwatch #" << i << " does not exist." << std::endl;
		} else {
			dur_store[i] = dur_store[i].zero();
			is_running[i] = false;
		}
	}

	void reset(std::string const& info) {
		reset(0, info);
	}

	template <typename F, typename ...Args>
	void_t< return_t<F,Args...> > timeit(std::string const& info, unsigned int const& N, F f, Args const& ...args) {
		dur_t dur;
		iclock::time_point start = iclock::now();
		try_it<F, Args...>(N, f, args...);
		dur = iclock::now() - start;
		if (!info.empty())
			std::cout << info << ": ";
		std::cout << "average elapsed time for " << N << " trials = " << dur.count() / ( N ? N : 1 ) << " seconds." << std::endl;
	}

	template <typename F, typename ...Args>
	void_t< return_t<F,Args...> > timeit(std::string const& info, F f, Args const& ...args) {
		timeit(info, 10u, f, args...);
	}

	template <typename F, typename ...Args>
	void_t< return_t<F,Args...> > timeit(unsigned int const& N, F f, Args const& ...args) {
		timeit(std::string(""), N, f, args...);
	}

	template <typename F, typename ...Args>
	void_t< return_t<F,Args...> > timeit(F f, Args const& ...args) {
		timeit(std::string(""), 10u, f, args...);
	}

	template <typename F, typename ...Args>
	void timeit(...) {
		std::cerr << "Stopwatch error: timeit: invalid call." << std::endl;
	}
};


// remove the beginning and trailing whitespaces/tabs of a string
std::string trim(std::string const& str, std::string whitespace =" \t") {
	std::string out = str;
	auto start = str.find_first_not_of(whitespace);
	if (start == std::string::npos)
		return "";
	auto end = str.find_last_not_of(whitespace);
	auto range = end - start + 1;
	return str.substr(start, range);
}

// check if a string starts with a certain string
bool start_with(std::string const& pre, std::string const& str) {
	return (std::strncmp(pre.c_str(), str.c_str(), pre.size()) == 0);
}

// keyword parser
struct Parser
{
	Parser(std::vector<std::string> const& keys_) : keys(keys_), vals(keys.size()) {}

	std::vector<std::string> keys;
	std::vector<std::string> vals;

	void reset(std::vector<std::string> const& keys_) {
		keys = keys_;
		vals = std::vector<std::string>(keys.size(), "");
	}

	void parse(std::string const& file) {
		std::fstream fs(file);
		std::string line, str;
		std::stringstream ss;
		while (std::getline(fs, line)) {
			ss << line;
			while (std::getline(ss, str, ',')) {
				auto start = str.find_first_not_of(" \t");
				str.erase(0, start);
				for (size_t i = 0; i != keys.size(); ++i) {
					if (start_with(keys[i], str)) {
						str.erase(0, keys[i].size());
						vals[i] = trim(str);
						break;
					}
				}
			}
			ss.str("");
			ss.clear();
		}
	}

	template <int N = 0>
	void pour(std::string& val) {
		size_check<N>();
		val = vals[N];
	}

	template <int N = 0, typename T>
	void pour(T& val) {
		size_check<N>();
		std::stringstream ss;
		ss << vals[N];
		ss >> val;
	}

	template <int N = 0, typename T, typename ...Rs>
	void pour(T& val, Rs& ...args) {
		pour<N>(val);
		pour<N+1>(args...);
	}


	private:

	template <int N>
	void size_check() {
		if ( N >= keys.size() ) {
			std::cerr << "Parser error: too many variables: expect " << keys.size() << " or less." << std::endl;
			exit(EXIT_FAILURE);
		}
	}
};

/*
template <typename ...Ts>
struct Parser
{
	Parser(std::vector<std::string> const& keys_) : keys(keys_), vals(keys.size()) {
		if ( sizeof...(Ts) > keys.size() ) {
			std::cerr << "Parser error: too many types specified." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	void reset(std::vector<std::string> const& keys_) {
		keys = keys_;
		vals = std::vector<std::string>(keys.size(), "");
	}

	std::vector<std::string> keys;
	std::vector<std::string> vals;

	void parse(std::string const& file) {
		std::fstream fs(file);
		std::string line;
		std::string str, first_word;
		std::stringstream ss;
		while (std::getline(fs, line)) {
			ss << line;
			while (std::getline(ss, str, ',')) {
				auto start = str.find_first_not_of(" \t");
				str.erase(0, start);
				auto stop = str.find_first_of(" \t");
				first_word = str.substr(0, stop);
				for (size_t i = 0; i != keys.size(); ++i) {
					if (first_word == keys[i]) {
						str.erase(0, keys[i].length());
						vals[i] = trim(str);
						break;
					}
				}
			}
			ss.str("");
			ss.clear();
		}
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


	private:

	std::string trim(std::string const& str, std::string whitespace =" \t") {
		std::string out = str;
		auto start = str.find_first_not_of(whitespace);
		if (start == std::string::npos)
			return "";
		auto end = str.find_last_not_of(whitespace);
		auto range = end - start + 1;
		return str.substr(start, range);
	}

	template <int N, typename T>
	typename std::enable_if< ( N >= sizeof...(Ts) ), void>::type type_check() {}

	template <int N, typename T>
	typename std::enable_if< ( N < sizeof...(Ts) ), void>::type type_check() {
		using C = typename std::tuple_element<N, std::tuple<Ts...>>::type;
		if ( !std::is_same<void, C>::value && !std::is_same<T, C>::value ) {
			std::cerr << "Parser error: type mismatch: argument " << N << std::endl
				<< "expected type: " << typeid(C).name() << std::endl
				<< "detected type: " << typeid(T).name() << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	template <int N>
	void size_check() {
		if ( N >= keys.size() ) {
			std::cerr << "Parser error: too many variables." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	template <int N, typename T>
	void check() {
		size_check<N>();
		type_check<N,T>();
	}

};
*/


#endif
