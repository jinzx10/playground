/* numerical gradient of real functions by finite difference
 */

#ifndef __GRADIENT_H__
#define __GRADIENT_H__

#include <functional>
#include <armadillo>

inline std::function<double(double)> grad(std::function<double(double)> const& f) {
	double delta = 0.001;
	return [f, delta] (double x) -> double {
		return ( -f(x-3.0*delta)/60.0 + 3.0*f(x-2.0*delta)/20.0 - 3.0*f(x-delta)/4.0 +
				f(x+3.0*delta)/60.0 - 3.0*f(x+2.0*delta)/20.0 + 3.0*f(x+delta)/4.0 ) 
			/ delta;
	};
}

inline arma::vec pt(arma::vec x, arma::uword const& i, double d) {
	x(i) += d;
	return x;
}

inline std::function<double(arma::vec)> gradi(std::function<double(arma::vec)> const& f, arma::uword const& i) {
	double delta = 0.001;
	return [f, i, delta] (arma::vec const& x) -> double {
		return ( -f(pt(x,i,-3.0*delta))/60.0 + 3.0*f(pt(x,i,-2.0*delta))/20.0 - 3.0*f(pt(x,i,-delta))/4.0 +
				f(pt(x,i,3.0*delta))/60.0 - 3.0*f(pt(x,i,2.0*delta))/20.0 + 3.0*f(pt(x,i,delta))/4.0 ) 
			/ delta;
	};
}

inline std::function<arma::vec(arma::vec)> grad(std::function<double(arma::vec)> const& f) {
	return [f] (arma::vec const& x) -> arma::vec {
		arma::vec df = arma::zeros(x.n_elem);
		for (arma::uword i = 0; i != x.n_elem; ++i) {
			df(i) = gradi(f,i)(x);
		}
		return df;
	};
}

#endif
