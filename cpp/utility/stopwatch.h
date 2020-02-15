#ifndef __STOPWATCH_H__
#define __STOPWATCH_H__

#include <iostream>
#include <chrono>
#include <string>

struct Stopwatch
{
	using iclock	= std::chrono::high_resolution_clock;
	using dur_t		= std::chrono::duration<double>;

	Stopwatch(): t_start(), dur_store(dur_t::zero()), is_running(false) {}

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
		std::cout << "time elapsed = " << dur.count() << " seconds" << std::endl; 
	}

	void reset(std::string const& info = "") { 
		if (!info.empty())
			std::cout << info << std::endl;
		dur_store = dur_store.zero();
		is_running = false;
	}

	/*
	template <typename F, typename...Args>
	void timeit(F f, Args& ...args) {
		iclock::time_point start = iclock::now();
		dur_t dur;
		auto result = f(args...);
		dur = iclock::now() - start;
		std::cout << "time elapsed = " << dur.count() << std::endl;
	}
	*/
	template <int N = 10, typename F, typename...Args>
	typename std::enable_if<N!=0, void>::type timeit(F f, Args& ...args) {
		iclock::time_point start = iclock::now();
		dur_t dur;
		auto result = f(args...);
		timeit<N-1, F, Args...>(f, args...);
		dur = iclock::now() - start;
		std::cout << "time elapsed = " << dur.count() << std::endl;
	}

	template <typename ...Ts>
	void timeit(...) {}

	iclock::time_point t_start;	
	dur_t dur_store;
	bool is_running;
};


#endif
