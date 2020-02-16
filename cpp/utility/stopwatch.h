#ifndef __STOPWATCH_H__
#define __STOPWATCH_H__

#include <iostream>
#include <chrono>
#include <string>

struct Stopwatch
{
	using iclock	= std::chrono::high_resolution_clock;
	using dur_t		= std::chrono::duration<double>;

	template <typename ...>
	using void_t	= void;

	template <typename F, typename ...Args>
	using return_t	= decltype( std::declval<F>()(std::declval<Args>()...) );

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


	private:

	template <typename F, typename ...Args>
	void try_it(unsigned int const& N, F f, Args const& ...args) {
		if (N != 0) {
			f(args...);
			try_it(N-1, f, args...);
		}
	}
};

#endif
