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
	using iclock	= std::chrono::high_resolution_clock;
	using dur_t		= std::chrono::duration<double>;

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

	Stopwatch(): t_start(), dur_store(dur_t::zero()), is_running(false) {}

	iclock::time_point t_start;	
	dur_t dur_store;
	bool is_running;

	void run(std::string const& info = "") {
		if (!info.empty())
			std::cout << info << std::endl;
		if (is_running) {
			std::cout << "The stopwatch is already running. Nothing to do." << std::endl;
		} else {
			t_start = iclock::now();
			is_running = true;
		}
	}

	void pause(std::string const& info = "") { 
		if (!info.empty())
			std::cout << info << std::endl;
		if (is_running) {
			dur_store += iclock::now() - t_start;
			is_running = false;
		} else {
			std::cout << "The stopwatch is not running. Nothing to do." << std::endl;
		}
	}

	void report(std::string const& info = "") { 
		if (!info.empty())
			std::cout << info << ": "; 
		dur_t dur = is_running ? dur_store + static_cast<dur_t>(iclock::now() - t_start) : dur_store;
		std::cout << "elapsed time = " << dur.count() << " seconds" << std::endl; 
	}

	void reset(std::string const& info = "") { 
		if (!info.empty())
			std::cout << info << std::endl;
		dur_store = dur_store.zero();
		is_running = false;
	}

	template <typename F, typename ...Args>
	void_t< return_t<F,Args...> > timeit(std::string const& info, unsigned int const& N, F f, Args const& ...args) {
		dur_t dur;
		iclock::time_point start = iclock::now();
		try_it<F, Args...>(N, f, args...);
		dur = iclock::now() - start;
		if (!info.empty())
			std::cout << info << ": ";
		std::cout << "average elapsed time for " << N << " trials = " << dur.count() / ( N ? N : 1 ) << " seconds" << std::endl;
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
};


// keyword parser
template <typename ...Ts>
struct Parser
{
	Parser(std::vector<std::string> const& keys_) : keys(keys_), vals(keys.size()) {
		if ( sizeof...(Ts) > keys.size() ) {
			std::cerr << "Parser error: too many types specified." << std::endl;
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
			ss << line;
			while (std::getline(ss, str, ',')) {
				for (size_t i = 0; i != keys.size(); ++i) {
					auto pos = str.find(keys[i]);
					if (pos != std::string::npos) {
						str.erase(pos, keys[i].length());
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



#endif
