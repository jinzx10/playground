#ifndef __STOPWATCH_H__
#define __STOPWATCH_H__

#include <iostream>
#include <chrono>

class Stopwatch
{
	public:
		using iclock	= std::chrono::high_resolution_clock;
		using dur_t		= std::chrono::duration<double>;

		Stopwatch(): t_start(), dur_store(dur_t::zero()), is_running(false) {}

		void run() {
			if (is_running) {
				std::cout << "The stopwatch is already running. Nothing done." << std::endl;
			} else {
				t_start = iclock::now();
				is_running = true;
			}
		}

		void report() { 
			dur_t dur = is_running ? dur_store + static_cast<dur_t>(iclock::now() - t_start) : dur_store;
			std::cout << "time elapsed = " << dur.count() << " seconds" << std::endl; 
		}

		void reset() { 
			dur_store = dur_store.zero();
			is_running = false;
		}

		void pause() { 
			if (is_running) {
				dur_store += iclock::now() - t_start;
				is_running = false;
			} else {
				std::cout << "The stopwatch is not running. Nothing done." << std::endl;
			}
		}

	private:
		iclock::time_point t_start;	
		dur_t dur_store;
		bool is_running;
};

#endif