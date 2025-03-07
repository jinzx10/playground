#include <cstdio>
#include <cmath>
#include <cassert>
#include <array>
#include <algorithm>

const double pi = std::acos(-1.0);
const double root2 = std::sqrt(2.0);

int minus_1_pow(int m) {
	return m % 2 ? -1 : 1;
}


double factorial(const int n) {
    assert( n >= 0 );
    double val = 1.0;
    for(int i = 2; i <= n; i++)
    {
        val *= static_cast<double>(i);
    }
    return val;
}


bool is_valid_lm(const int l1, const int l2, const int l3, const int m1, const int m2, const int m3) {
    return std::abs(m1) <= l1 && std::abs(m2) <= l2 && std::abs(m3) <= l3;
}


bool gaunt_select_l(const int l1, const int l2, const int l3) {
	// applies to both standard Gaunt & real Gaunt coefficients
    return l1 + l2 >= l3 && l1 + l3 >= l2 && l2 + l3 >= l1 && (l1 + l2 + l3) % 2 == 0;
}


bool gaunt_select_m(const int m1, const int m2, const int m3) {
	// applies to standard Gaunt coefficients only
	return m1 + m2 + m3 == 0;
}


double gaunt(const int l1, const int l2, const int l3, const int m1, const int m2, const int m3) {

    // This function computes the Gaunt coefficients from the Wigner-3j expression

    assert( is_valid_lm(l1, l2, l3, m1, m2, m3) );

    if ( !gaunt_select_l(l1, l2, l3) || !gaunt_select_m(m1, m2, m3) )
    {
        return 0.0;
    }

    int g = (l1 + l2 + l3) / 2;
    double pref = std::sqrt( (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4.0 * pi) );
    double tri = std::sqrt( factorial(l1 + l2 - l3) * factorial(l2 + l3 - l1) * factorial(l3 + l1 - l2)
                            / factorial(l1 + l2 + l3 + 1) );

    // wigner3j(l1,l2,l3,0,0,0)
    double wigner1 = minus_1_pow(g) * tri * factorial(g) / factorial(g - l1) / factorial(g - l2) / factorial(g - l3);

    // wigner3j(l1,l2,l3,m1,m2,m3)
    int kmin = std::max(l2 - l3 - m1, l1 - l3 + m2);
    kmin = std::max(kmin, 0);

    int kmax = std::min(l1 - m1, l2 + m2);
    kmax = std::min(kmax, l1 + l2 - l3);

    double wigner2 = 0.0;
    for (int k = kmin; k <= kmax; ++k)
    {
        wigner2 += minus_1_pow(k) / factorial(k) / factorial(l1 - m1 - k) / factorial(l2 + m2 - k)
            / factorial(l3 - l2 + m1 + k) / factorial(l3 - l1 - m2 + k) / factorial(l1 + l2 - l3 - k);
    }

    wigner2 *= tri * minus_1_pow(l1 - l2 - m3) * std::sqrt(
            factorial(l1 + m1) * factorial(l1 - m1) *
            factorial(l2 + m2) * factorial(l2 - m2) *
            factorial(l3 + m3) * factorial(l3 - m3) );

    return pref * wigner1 * wigner2;
}


bool real_gaunt_select_m(const int m1, const int m2, const int m3) {
    return  ( ( (m1 < 0) + (m2 < 0) + (m3 < 0) ) % 2 == 0 ) &&
            ( std::abs(m1) + std::abs(m2) == std::abs(m3) ||
              std::abs(m2) + std::abs(m3) == std::abs(m1) ||
              std::abs(m3) + std::abs(m1) == std::abs(m2) );
}


double real_gaunt(const int l1, const int l2, const int l3, const int m1, const int m2, const int m3) {

    // This function calculates and returns the Gaunt coefficients of real spherical harmonics
	// Note that the sign convention of real spherical harmonics assumed here differs from the
	// standard definition by (-1)^m !!!

    assert( is_valid_lm(l1, l2, l3, m1, m2, m3) );

    if ( !gaunt_select_l(l1, l2, l3) || !real_gaunt_select_m(m1, m2, m3) )
    {
        return 0.0;
    }

    std::array<int, 3> m = {std::abs(m1), std::abs(m2), std::abs(m3)};
    int& m_absmax = *std::max_element(m.begin(), m.end());

    if ( m1 == 0 || m2 == 0 || m3 == 0 )
    {
        m_absmax = -m_absmax;
        return minus_1_pow(m_absmax) * gaunt(l1, l2, l3, m[0], m[1], m[2]);
    }
    else if ( m1 + m2 + m3 == 0 )
    {
        return root2 / 2.0 * minus_1_pow(m_absmax + 1) * gaunt(l1, l2, l3, m1, m2, m3);
    }
    else
    {
        m_absmax = -m_absmax;
        return root2 / 2.0 * minus_1_pow(m_absmax) * gaunt(l1, l2, l3, m[0], m[1], m[2]);
    }
}


int main() {

	int lmax = 2;
	for (int l1 = 0; l1 <= lmax; ++l1) {
		for (int l2 = 0; l2 <= lmax; ++l2) {
			for (int l3 = 0; l3 <= 2*lmax; ++l3) {
				for (int m1 = -l1; m1 <= l1; ++m1) {
					for (int m2 = -l2; m2 <= l2; ++m2) {
						for (int m3 = -l3; m3 <= l3; ++m3) {
							double tmp = real_gaunt(l1, l2, l3, m1, m2, m3);
							if (tmp != 0.0) {
								printf("l1=%1i  l2=%1i  l3=%1i  m1=% 2i  m2=% 2i  m3=% 2i  G=%20.15f\n",
										l1, l2, l3, m1, m2, m3, tmp);
								//printf("l1=%1i  l2=%1i  l3=%1i  m1=% 2i  m2=% 2i  m3=% 2i  G=%20.15f   %20.15f\n",
								//		l1, l2, l3, m1, m2, m3, tmp, tmp*minus_1_pow(m1)*minus_1_pow(m2)*minus_1_pow(m3));
							}
						}
					}
				}
			}
		}
	}

	return 0;
}


