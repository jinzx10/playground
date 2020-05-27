#ifndef __MATH_HELPER_H__
#define __MATH_HELPER_H__

#include <armadillo>
#include "arma_helper.h"
#include <functional>
#include <iostream>
#include <tuple>

// return the x and p standard deviation 
// of the Wigner quasiprobability distribution of simple harmonic oscillators
inline std::tuple<double, double> ho_wigner(double const& mass, double const& omega, double const& kT = 0) {
	double sigma_x, sigma_p;
	if (kT < arma::datum::eps) {
		sigma_x = std::sqrt(0.5 / mass / omega);
		sigma_p = std::sqrt(0.5 * omega * mass);
	} else {
		sigma_x = std::sqrt( 0.5 / mass / omega / std::tanh(omega/2.0/kT) );
		sigma_p = std::sqrt( 0.5 * mass * omega / std::tanh(omega/2.0/kT) );
	}
	return std::make_tuple(sigma_x, sigma_p);
}


// Fermi function
inline double fermi(double const& E, double const& mu, double const& kT) {
	return ( std::abs(kT) < arma::datum::eps ) ? 
		(E < mu) : 1.0 / ( std::exp( (E - mu) / kT ) + 1.0 );
}

inline arma::vec fermi(arma::vec const& E, double const& mu, double const& kT) {
	return ( std::abs(kT) < arma::datum::eps ) ? 
		arma::conv_to<arma::vec>::from(E < mu) : 1.0 / ( exp( (E - mu) / kT ) + 1.0 );
}

// Boltzmann weight
inline arma::vec boltzmann(arma::vec const& E, double const& kT) {
	arma::uword imin = E.index_min();
	return ( std::abs(kT) < arma::datum::eps ) ?
		unit_vec(E.n_elem, imin) : arma::exp(-(E-E(imin))/kT) / arma::accu( arma::exp(-(E-E(imin))/kT) );
}

// Gaussian
inline double gauss(double const& x, double const& mu, double const& sigma) {
	return 1.0 / sigma / sqrt( 2.0 * arma::datum::pi ) 
		* exp( -(x-mu)*(x-mu) / 2.0 / sigma / sigma );
}

inline arma::mat gauss(arma::vec const& x, arma::rowvec const& y, double const& sigma) {
	return exp( -0.5*arma::square( bcast_minus(x, y) / sigma ) ) 
		/ ( sigma * sqrt(2.0 * arma::datum::pi) );
}


// find the smallest/largest number
inline double min(double const& i) {
	return i;
}

template <typename ...Ts>
double min(double const& i, Ts const& ...args) {
    double tmp = min(args...);
    return ( i < tmp ) ? i : tmp;
}

inline double max(double const& i) {
	return i;
}

template <typename ...Ts>
double max(double const& i, Ts const& ...args) {
    double tmp = max(args...);
    return ( i > tmp ) ? i : tmp;
}


// numerical gradient of real functions by finite difference
inline std::function<double(double)> grad(std::function<double(double)> const& f, double const& delta = 0.001) {
	return [=] (double x) -> double {
		return ( -f(x-3.0*delta)/60.0 + 3.0*f(x-2.0*delta)/20.0 - 3.0*f(x-delta)/4.0 +
				f(x+3.0*delta)/60.0 - 3.0*f(x+2.0*delta)/20.0 + 3.0*f(x+delta)/4.0 ) 
			/ delta;
	};
}

template <typename V>
std::function<double(V)> gradi(std::function<double(V)> const& f, size_t const& i, double const& delta = 0.001) {
	return [=] (V const& v) -> double {
		std::function<double(double)> g = [=, v=v] (double const& x) mutable {
			v[i] = x;
			return f(v);
		};
		return grad(g, delta)(v[i]); 
	};
}

template <typename V>
std::function<V(V)> grad(std::function<double(V)> const& f, double const& delta = 0.001) {
	return [=] (V const& x) -> V {
		V df = x;
		for (size_t i = 0; i != x.size(); ++i) {
			df[i] = gradi(f, i, delta)(x);
		}
		return df;
	};
}


// generate grid points according to some grid density
inline arma::vec grid(double const& xmin, double const& xmax, std::function<double(double)> density) {
	arma::vec g = {xmin};
	double g_last = g.back();
	while ( g_last < xmax ) {
		g.resize(g.n_elem+1);
		g.back() = g_last + 1.0 / density(g_last);
		g_last = g.back();
	}
	return g;
}


// Newton-Raphson root-finding algorithm
// use finite difference to calculate the Jacobian
inline int newtonroot(std::function<double(double)> f, double& x, double const& dx = 1e-6, double const& tol = 1e-12, unsigned int const& max_iter = 50) {
	double fx = 0;
	double J = 0;
	for (unsigned int counter = 0; counter != max_iter; ++counter) {
		fx = f(x);
		if ( std::abs(fx) < tol )
			return 0;
		J = ( f(x+dx) - fx ) / dx;
		x -= fx / J;
	}
	return 1;
}

inline int newtonroot(std::function<arma::vec(arma::vec)> f, arma::vec& x, double const& dx = 1e-6, double const& tol = 1e-12, unsigned int const& max_iter = 50) {
	arma::uword len_x = x.n_elem;
	arma::vec dxi(len_x);
	arma::vec fx;
	arma::mat J;
	for (unsigned int counter = 0; counter != max_iter; ++counter) {
		fx = f(x);
		if ( norm(fx) < tol )
			return 0;
		J.set_size(fx.n_elem, len_x);
		for (arma::uword i = 0; i != len_x; ++i) {
			dxi.zeros();
			dxi(i) = dx;
			J.col(i) = ( f(x+dxi) - fx ) / dx;
		}
		x -= arma::solve(J, fx);
	}
	return 1;
}

// Broyden's quasi-Newton method
// the good and bad Broyden's methods are identical in 1D
inline int broydenroot(std::function<double(double)> f, double& x, double const& tol = 1e-12, unsigned int const& max_iter = 50) {
	double fx = f(x);
	if (std::abs(fx) < tol)
		return 0;

	// compute the initial Jacobian by finite difference
	double delta = 1e-6 * std::max(1.0, std::sqrt(std::abs(x)));
	double J = (f(x+delta) - fx) / delta; 

	double dx = 0.0;
	double fx_new = 0.0;
	for (unsigned int counter = 1; counter != max_iter; ++counter) {
		if (std::abs(J) < 1e-14) {
			std::cout << "broydenroot: the Jacobian appears to be singular." << std::endl;
			return 2;
		}

		dx = -fx / J;
		x += dx;
		fx_new = f(x);
		if (std::abs(fx_new) < tol)
			return 0;

		J = (fx_new - fx) / dx;
		fx = fx_new;
	}

	std::cout << "broydenroot: fails to find the root." << std::endl;
	return 1;
}

inline int broydenroot(std::function<arma::vec(arma::vec)> f, arma::vec& x, double const& tol = 1e-12, unsigned int const& max_iter = 50, std::string const& method = "good") {
	arma::vec fx = f(x);
	if (arma::norm(fx) < tol)
		return 0;

	int md = -1;
	if (!method.compare("good"))
		md = 0;
	if (!method.compare("inv"))
		md = 1;
	if (!method.compare("bad"))
		md = 2;

	if ( md < 0 ) {
		std::cerr << "broydenroot: invalid method" << std::endl;
		exit(EXIT_FAILURE);
	}

	arma::uword len_x = x.n_elem;
	arma::uword len_f = fx.n_elem;

	// compute the initial Jacobian by finite difference
	arma::mat J(len_f, len_x);
	double delta = 1e-6 * max(1.0, std::sqrt(arma::norm(x)));
	arma::vec dxi(len_x);
	arma::vec df(len_f);
	for (arma::uword i = 0; i != len_x; ++i) {
		dxi.zeros();
		dxi(i) = delta;
		df = f(x+dxi) - fx;
		J.col(i)  = df / delta;
	}

	arma::mat invJ;
	if ( md > 0 ) {
		if (len_x == len_f) {
			bool info = arma::inv(invJ, J);
			if (!info) {
				std::cout << "broydenroot: the Jacobian appears to be singular." << std::endl;
				return 2;
			}
		} else {
			std::cerr << "broydenroot: inverse update requires the number of equations equals to the number of variables." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	arma::vec dx(len_x);
	arma::vec fx_new(len_f);
	for (unsigned int counter = 1; counter != max_iter; ++counter) {
		if (md > 0) {
			dx = -invJ * fx;
		} else {
			dx = -arma::solve(J, fx);
		}

		x += dx;
		fx_new = f(x);

		if (arma::norm(fx_new) < tol)
			return 0;

		df = fx_new - fx;
		fx = fx_new;

		// Broyden's update
		switch (md) {
			case 2:
				invJ += ( dx - invJ * df ) / arma::dot(df, df) * df.as_row();
				break;
			case 1:
				invJ += ( dx - invJ * df ) / arma::dot(dx, invJ * df) * dx.as_row() * invJ;
				break;
			default:
				J += ( df - J * dx ) / arma::dot(dx, dx) * dx.as_row();
		}
	}

	std::cout << "broydenroot: fails to find the root." << std::endl;
	return 1;
}


// solve for the chemical potential given the particle number and temperature
inline int findmu(double& mu, arma::vec const& E, arma::uword const& n, double const& kT = 0.0) {
	if ( n > E.n_elem ) {
		std::cerr << "findmu: there are more particles than energy levels." << std::endl;
		exit(EXIT_FAILURE);
	}

	if ( std::abs(kT) < arma::datum::eps ) {
		mu = arma::sort(E).eval()(n-1);
		return 0;
	}

	auto dn = [&] (double const& mu) { return arma::accu(fermi(E, mu, kT)) - n; };
	mu = E(0);
	return broydenroot(dn, mu, 1e-14);
}


// linear interpolation (or extrapolation, if outside the range)
inline double lininterp(double const& x0, arma::vec const& x, arma::vec const& y, bool is_evenly_spaced = false) {
	arma::uword i = 1;

	// x must be sorted in ascending order
	if ( is_evenly_spaced ) {
		if ( x0 > x(0) && x0 < x(x.n_elem-1) ) {
			i = (x0 - x(0)) / (x(1) - x(0)) + 1; 
		} else {
			if ( x0 >= x(x.n_elem-1) )
				i = x.n_elem - 1;
		}
	} else {
		// must not have repeated elements
		for (i = 1; i != x.n_elem-1; ++i)
			if ( x(i) > x0 ) break;

	}

	return y(i-1) + ( y(i) - y(i-1) ) / ( x(i) - x(i-1) ) * (x0 - x(i-1));
}

inline arma::rowvec lininterp(double const& x0, arma::vec const& x, arma::mat const& y, bool is_evenly_spaced = false) {
	arma::uword i = 1;

	// x must be sorted in ascending order
	if ( is_evenly_spaced ) {
		if ( x0 > x(0) && x0 < x(x.n_elem-1) ) {
			i = (x0 - x(0)) / (x(1) - x(0)) + 1; 
		} else {
			if ( x0 >= x(x.n_elem-1) )
				i = x.n_elem - 1;
		}
	} else {
		// must not have repeated elements
		for (i = 1; i != x.n_elem-1; ++i)
			if ( x(i) > x0 )
				break;
	}

	return y.row(i-1) + 
		( y.row(i) - y.row(i-1) ) / ( x(i) - x(i-1) ) * (x0 - x(i-1));
}


inline bool null_qr(arma::mat& ns, arma::mat const& A) {
    if (A.is_empty()) {
        ns.clear();
        return true;
    }

    arma::mat q, r;
    bool status = arma::qr(q, r, A.t());
    if (status) {
        arma::vec s = arma::sum(arma::abs(r), 1);
        double tol = arma::datum::eps * std::max(A.n_rows, A.n_cols);
        ns = q.cols(arma::find(s<tol));
    }
    return status;
}

inline arma::mat null_qr(arma::mat const& A) {
	arma::mat ns;
	bool status = null_qr(ns, A);
	if (!status) {
		std::cout << "null_qr(): qr decomposition failed." << std::endl;
		exit(EXIT_FAILURE);
	}
	return ns;
}

inline arma::mat orth_lowdin(arma::mat const& A) {
	arma::vec eigval;
	arma::mat eigvec;
	arma::eig_sym(eigval, eigvec, A*A.t());
	return arma::solve(eigvec*arma::diagmat(arma::sqrt(eigval))*eigvec.t(), A);
}


#endif
