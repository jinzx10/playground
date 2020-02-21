#ifndef __WIDGETS_H__
#define __WIDGETS_H__

#include <string>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

template <int N = 1, typename T>
void readargs(char** args, T& var) {
	std::stringstream ss;
	ss << args[N];
	ss >> var;
}

template <int N = 1, typename T, typename ...Ts>
void readargs(char** args, T& var, Ts& ...vars) {
	std::stringstream ss;
	ss << args[N];
	ss >> var;
	readargs<N+1>(args, vars...);
}


inline int mkdir(std::string const& dir) {
	std::string command = "mkdir -p " + dir;
	return std::system(command.c_str());
}


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


struct Parser
{
	Parser(std::vector<std::string> const& keys_) : keys(keys_), vals(keys.size()) {}

	std::vector<std::string> keys;
	std::vector<std::string> vals;

	void parse(std::string const& file) {
		std::fstream fs(file);
		std::string str;
		while (std::getline(fs, str)) {
			for (size_t i = 0; i != keys.size(); ++i) {
				auto pos = str.find(keys[i]);
				if (pos != std::string::npos) {
					str.erase(pos, keys[i].length());
					vals[i] = str;
					break;
				}
			}
		}
	}

	template <int N = 0, typename T>
	void pour(T& val) {
		if (N >= keys.size()) {
			std::cerr << "Too many variables." << std::endl;
			return;
		}
		std::stringstream ss;
		ss << vals[N];
		ss >> val;
	}

	template <int N = 0, typename T, typename ...Ts>
	void pour(T& val, Ts& ...args) {
		if (N >= keys.size()) {
			std::cerr << "Too many variables." << std::endl;
			return;
		}
		std::stringstream ss;
		ss << vals[N];
		ss >> val;
		pour<N+1, Ts...>(args...);
	}
};



#endif
