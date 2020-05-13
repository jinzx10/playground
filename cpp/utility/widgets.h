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
// no bound check!
template <int N = 1>
void readargs(char** args, std::string& var) {
	var = args[N];
}

template <int N = 1, typename T>
void readargs(char** args, T& var) {
	std::stringstream ss(args[N]);
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

// touch
inline int touch(std::string const& file) {
	std::string command = "touch " + file;
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


// remove the leading and trailing characters in char_rm of a string
// by default char_rm contains whitespace and tab
inline std::string trim(std::string const& str, std::string char_rm =" \t") {
	auto start = str.find_first_not_of(char_rm);
	return (start == std::string::npos) ? 
		"" : str.substr(start, str.find_last_not_of(char_rm)-start+1);
}

// check if a string starts with a certain string
// leading characters in char_skip will be ignored
inline bool start_with(std::string const& pre, std::string const& str, std::string char_skip = "") {
	auto start = str.find_first_not_of(char_skip);
	return (start == std::string::npos) ? (pre.size() ? false : true) : 
		(std::strncmp(pre.c_str(), str.substr(start).c_str(), pre.size()) == 0);
}

// replace the leading "~" of a string with ${HOME}
// leading and trailing characters in char_skip (whitespace and tab by default) will be ignored
inline std::string expand_leading_tilde(std::string const& dir, std::string char_skip = " \t") {
	auto start = dir.find_first_not_of(char_skip);
	return (start == std::string::npos || dir[start] != '~') ? dir :
		dir.substr(0, start) + std::getenv("HOME") + dir.substr(start+1);;
}

// keyword parser
// basic usage: Parser p({"key1", "key2", ...}); p.parse(file); p.pour(val1, val2, ...);
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


#endif
