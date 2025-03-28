#include <cmath>
#include <cassert>
#include <vector>
#include <cstdio>
#include <complex>

const int LMAX = 4;


double pm1(int m) {
	return m % 2 ? -1 : 1;
}


double factorial(int n)
{
    assert( n >= 0 );
    double val = 1.0;
    for(int i = 2; i <= n; i++)
    {
        val *= static_cast<double>(i);
    }
    return val;
}


int pack_lm(int l, int m)
{
    assert( std::abs(m) <= l );
    return l * l + l + m;
}


bool is_valid_lm(int l1, int l2, int l3, int m1, int m2, int m3)
{
    return std::abs(m1) <= l1 && std::abs(m2) <= l2 && std::abs(m3) <= l3;
}


bool select_l(int l1, int l2, int l3)
{
    return l1 + l2 >= l3 && l1 + l3 >= l2 && l2 + l3 >= l1 && (l1 + l2 + l3) % 2 == 0;
}


bool select_m(int m1, int m2, int m3) {
	return m1 + m2 + m3 == 0;
}


//bool select_m_real(int m1, int m2, int m3)
//{
//    return  ( ( (m1 < 0) + (m2 < 0) + (m3 < 0) ) % 2 == 0 ) &&
//            ( std::abs(m1) + std::abs(m2) == std::abs(m3) || 
//              std::abs(m2) + std::abs(m3) == std::abs(m1) || 
//              std::abs(m3) + std::abs(m1) == std::abs(m2) );
//}


double gaunt(int l1, int l2, int l3, int m1, int m2, int m3)
{
    // This function computes the Gaunt coefficients from the Wigner-3j expression

    assert( is_valid_lm(l1, l2, l3, m1, m2, m3) );
    if ( !select_l(l1, l2, l3) || !select_m(m1, m2, m3) )
    {
        return 0.0;
    }

    int g = (l1 + l2 + l3) / 2;
    double pref = std::sqrt( (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4.0*M_PI));
    double tri = std::sqrt( factorial(l1 + l2 - l3) * factorial(l2 + l3 - l1) * factorial(l3 + l1 - l2)
                            / factorial(l1 + l2 + l3 + 1) );

    // wigner3j(l1,l2,l3,0,0,0)
    double wigner1 = pm1(g) * tri * factorial(g) / factorial(g - l1) / factorial(g - l2) / factorial(g - l3);

    // wigner3j(l1,l2,l3,m1,m2,m3)
    int kmin = std::max(l2 - l3 - m1, l1 - l3 + m2);
    kmin = std::max(kmin, 0);

    int kmax = std::min(l1 - m1, l2 + m2);
    kmax = std::min(kmax, l1 + l2 - l3);

    double wigner2 = 0.0;
    for (int k = kmin; k <= kmax; ++k)
    {
        wigner2 += pm1(k) / factorial(k) / factorial(l1 - m1 - k) / factorial(l2 + m2 - k)
            / factorial(l3 - l2 + m1 + k) / factorial(l3 - l1 - m2 + k) / factorial(l1 + l2 - l3 - k);
    }

    wigner2 *= tri * pm1(l1 - l2 - m3) * std::sqrt(
            factorial(l1 + m1) * factorial(l1 - m1) *
            factorial(l2 + m2) * factorial(l2 - m2) *
            factorial(l3 + m3) * factorial(l3 - m3) );

    return pref * wigner1 * wigner2;
}

int gind(int l1, int m1, int l2, int m2, int q) {
	return (pack_lm(l1, m1) * (LMAX+1)*(LMAX+1) + pack_lm(l2, m2)) * (LMAX+1) + q;
}


void gaunt_gen(std::vector<double>& coef) {
	// Generate a table of Gaunt coefficients for the product of
	// spherical harmonics up to LMAX.
	// The index mapping of this table is handled by gind(), which
	// effectively interprets this vector as a C-style 3-d array
	//
	//		(l1,m1) x (l2,m2) x q
	//
	// where q = (l1+l2-l3)/2.
	// (The selection rule for m is incorporated.)
	for (int l1 = 0; l1 <= LMAX; ++l1) {
		for (int m1 = -l1; m1 <= l1; ++m1) {
			for (int l2 = 0; l2 <= LMAX; ++l2) {
				for (int m2 = -l2; m2 <= l2; ++m2) {
					for (int l3 = abs(l1-l2); l3 <= l1+l2; l3 += 2) {
						int m3 = m1 + m2;
						if (abs(m3) > l3) {
							continue;
						}
						int q = (l1+l2-l3)/2;
						double fac = std::sqrt(4.0*M_PI*(2*l3+1)/((2*l1+1)*(2*l2+1)));
						double G = fac * pm1(m3) * gaunt(l1, l2, l3, m1, m2, -m3);
						coef[gind(l1, m1, l2, m2, q)] = G;
						//printf("l1=%i  m1=%2i  l2=%i  m2=%2i  q=%i  G=%8.5f\n", l1, m1, l2, m2, q, G);
					}
				}
			}
		}
	}
}


std::complex<double> sph_harm(int l, int m, double theta, double phi) {
	int mabs = std::abs(m);
	auto Ylm = std::sph_legendre(l, mabs, theta) * std::exp(std::complex<double>(0, mabs*phi));
	if (m < 0) {
		Ylm = pm1(m) * std::conj(Ylm);
	}
	return Ylm;
}

std::complex<double> solid_harm(int l, int m, double x, double y, double z) {
	double rabs = std::sqrt(x*x + y*y + z*z);
	double theta = std::acos(z/rabs);
	double phi = std::atan2(y, x);
	return std::sqrt(4.0*M_PI/(2*l+1)) * std::pow(rabs,l) * sph_harm(l, m, theta, phi);
}


int binom(int n, int k) {
	if (n < k) {
		return 0;
  	} else if (n == k || k == 0 ) {
  	    return 1;
  	}
	return binom(n - 1, k) + binom(n - 1, k - 1);
}

int mind(int l, int m, int lp, int mp) {
	return pack_lm(l,m) * (LMAX+1)*(LMAX+1) + pack_lm(lp,mp);
}

void M_gen(std::vector<double>& coef) {
	for (int l = 0; l <= LMAX; ++l) {
		for (int m = -l; m <= l; ++m) {
			for (int lp = 0; lp <= l; ++lp) {
				for (int mp = std::max(-lp, m+lp-l); mp <= std::min(lp, m+l-lp); ++mp) {
					double M = std::sqrt(binom(l+m, lp+mp) * binom(l-m, lp-mp));
					coef[mind(l, m, lp, mp)] = M;
					//printf("l=%i  m=%2i  lp=%i  mp=%2i  M=%8.5f\n", l, m, lp, mp, M);
				}
			}
		}
	}
}


void MC(std::vector<double>& M, double ABx, double ABy, double ABz) {
	std::vector<std::complex<double>> CAB(std::pow(LMAX+1, 2));
	for (int l = 0; l <= LMAX; ++l) {
		for (int m = 0; m <= l; ++m) {
			auto val = solid_harm(l, m, ABx, ABy, ABz);
			CAB[pack_lm(l,m)] = val;
			CAB[pack_lm(l,-m)] = pm1(m) * std::conj(val);
		}
	}
}


int main() {

	std::vector<double> Gaunt(std::pow(LMAX+1, 5));
	std::vector<double> M    (std::pow(LMAX+1, 4));
	std::vector<double> MC0  (std::pow(LMAX+1, 4));
	std::vector<double> MC1  (std::pow(LMAX+1, 4));
	std::vector<double> MC2  (std::pow(LMAX+1, 4));

	gaunt_gen(Gaunt);
	M_gen(M);

	return 0;
}





