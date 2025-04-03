//#include <cmath>
#include <tr1/cmath>
#include <cassert>
#include <vector>
#include <cstdio>
#include <complex>
#include <random>
#include <chrono>

using iclock = std::chrono::high_resolution_clock;
std::chrono::duration<double> dur[10];

const int LMAX = 4;

// total number of (l,m) up to LMAX
const int SZ1 = (LMAX+1) * (LMAX+1);

// total number of (l,m) up to 2*LMAX
const int SZ2 = (2*LMAX+1) * (2*LMAX+1);

// total number of (q,l,m) in expansion
const int SZX = (LMAX+1) * SZ2;


double pm1(int m) {
    return m % 2 ? -1 : 1;
}


double factorial(int n) {
    assert( n >= 0 );
    double val = 1.0;
    for(int i = 2; i <= n; i++) {
        val *= static_cast<double>(i);
    }
    return val;
}


inline int pack_lm(int l, int m) {
    return l * l + l + m;
}


bool is_valid_lm(int l1, int l2, int l3, int m1, int m2, int m3) {
    return std::abs(m1) <= l1 && std::abs(m2) <= l2 && std::abs(m3) <= l3;
}


bool select_l(int l1, int l2, int l3) {
    return l1 + l2 >= l3 && l1 + l3 >= l2 && l2 + l3 >= l1 // triangle rule
        && (l1 + l2 + l3) % 2 == 0;
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


double gaunt(int l1, int l2, int l3, int m1, int m2, int m3) {
    // This function computes the Gaunt coefficients from the Wigner-3j expression
    // NOTE: this function is neither efficient nor accurate.
    // One may use sympy to get the exact expression, store them to file, and load
    // it when necessary.

    assert( is_valid_lm(l1, l2, l3, m1, m2, m3) );
    if ( !select_l(l1, l2, l3) || !select_m(m1, m2, m3) ) {
        return 0.0;
    }

    int g = (l1 + l2 + l3) / 2;
    double pref = std::sqrt( (2*l1 + 1) * (2*l2 + 1) * (2*l3 + 1) / (4.0*M_PI));
    double tri = std::sqrt( factorial(l1 + l2 - l3) * factorial(l2 + l3 - l1)
            * factorial(l3 + l1 - l2) / factorial(l1 + l2 + l3 + 1) );

    // wigner3j(l1,l2,l3,0,0,0)
    double wigner1 = pm1(g) * tri * factorial(g) / factorial(g - l1)
        / factorial(g - l2) / factorial(g - l3);

    // wigner3j(l1,l2,l3,m1,m2,m3)
    int kmin = std::max(l2 - l3 - m1, l1 - l3 + m2);
    kmin = std::max(kmin, 0);

    int kmax = std::min(l1 - m1, l2 + m2);
    kmax = std::min(kmax, l1 + l2 - l3);

    double wigner2 = 0.0;
    for (int k = kmin; k <= kmax; ++k) {
        wigner2 += pm1(k) / factorial(k) / factorial(l1 - m1 - k)
            / factorial(l2 + m2 - k) / factorial(l3 - l2 + m1 + k)
            / factorial(l3 - l1 - m2 + k) / factorial(l1 + l2 - l3 - k);
    }

    wigner2 *= tri * pm1(l1 - l2 - m3) * std::sqrt(
            factorial(l1 + m1) * factorial(l1 - m1) *
            factorial(l2 + m2) * factorial(l2 - m2) *
            factorial(l3 + m3) * factorial(l3 - m3) );

    return pref * wigner1 * wigner2;
}


inline int gind(int l1, int m1, int l2, int m2, int q) {
    return (pack_lm(l1, m1) * SZ1 + pack_lm(l2, m2)) * (LMAX+1) + q;
}


void gaunt_gen(std::vector<double>& coef) {
    // Generate a table of Gaunt coefficients for the product of
    // spherical harmonics up to LMAX.
    // The index mapping of this table is handled by gind(), which
    // effectively interprets this vector as a C-style 3-d array
    //
    //        (l1,m1) x (l2,m2) x q
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
                    }
                }
            }
        }
    }
}


std::complex<double> sph_harm(int l, int m, double theta, double phi) {
    int mabs = std::abs(m);
    auto Ylm = std::tr1::sph_legendre(l, mabs, theta)
                * std::exp(std::complex<double>(0, mabs*phi));
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
    } else if (n == k || k == 0) {
        return 1;
    }
    return binom(n - 1, k) + binom(n - 1, k - 1);
}


inline int pack_lmlm(int l, int m, int lp, int mp) {
    return pack_lm(l,m) * SZ1 + pack_lm(lp,mp);
}


void M_gen(std::vector<double>& coef) {
    for (int l = 0; l <= LMAX; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (int lp = 0; lp <= l; ++lp) {
                for (int mp = std::max(-lp, m+lp-l); mp <= std::min(lp, m+l-lp); ++mp) {
                    double M = std::sqrt(binom(l+m, lp+mp) * binom(l-m, lp-mp));
                    coef[pack_lmlm(l, m, lp, mp)] = M;
                }
            }
        }
    }
}


void MC0(std::vector<double> const& M, double ABx, double ABy, double ABz, std::vector<std::complex<double>>& mc0) {
    std::vector<std::complex<double>> CAB(SZ1);
    for (int l = 0; l <= LMAX; ++l) {
        for (int m = 0; m <= l; ++m) {
            auto val = solid_harm(l, m, ABx, ABy, ABz);
            CAB[pack_lm(l, m)] = val;
            CAB[pack_lm(l,-m)] = pm1(m) * std::conj(val);
        }
    }

    for (int i = 0; i < SZ1; ++i) {
        for (int j = 0; j < SZ1; ++j) {
            mc0[i*SZ1+j] = M[i*SZ1+j] * CAB[j];
        }
    }
}


void solid_harm_prod(std::vector<double> const& G, std::vector<std::complex<double>> const& mc0, double gamma, std::vector<std::complex<double>>& coef) {

    auto start = iclock::now();

    std::fill(coef.begin(), coef.end(), std::complex<double>(0,0));

    // mc0 times the exponent-dependent part
    std::vector<std::complex<double>> Mbar1(mc0.size()); 
    std::vector<std::complex<double>> Mbar2(mc0.size());

    double base1 = -1.0 / (1.0 + gamma);
    double base2 = 1.0 / (1.0 + 1.0/gamma);

    for (int l = 0; l <= LMAX; ++l) {
        double fac1 = 1.0;
        double fac2 = 1.0;
        for (int lpp = l; lpp >= 0; --lpp) {
            for (int m = -l; m <= l; ++m) {
                int lm = pack_lm(l,m);
                for (int mpp = std::max(-lpp, m+lpp-l); mpp <= std::min(lpp, m+l-lpp); ++mpp) {
                    int i = lm * SZ1 + pack_lm(lpp, mpp);
                    int j = lm * SZ1 + pack_lm(l-lpp, m-mpp);
                    Mbar1[i] = mc0[j] * fac1;
                    Mbar2[i] = mc0[j] * fac2;
                }
            }
            fac1 *= base1;
            fac2 *= base2;
        }
    }
                
    dur[2] += iclock::now() - start;

    start = iclock::now();

    for (int l1 = 0; l1 <= LMAX; ++l1) {
        for (int l2 = 0; l2 <= LMAX; ++l2) {
            for (int l1pp = 0; l1pp <= l1; ++l1pp) {
                for (int l2pp = 0; l2pp <= l2; ++l2pp) {
                    for (int m1 = -l1; m1 <= l1; ++m1) {
                        for (int m2 = -l2; m2 <= l2; ++m2) {
                            int ir = pack_lmlm(l1, m1, l2, m2);
                            for (int m1pp = m1 - std::min(l1-l1pp, m1+l1pp);
                                    m1pp <= m1 - std::max(l1pp-l1, m1-l1pp); ++m1pp) {
                                for (int m2pp = m2 - std::min(l2-l2pp, m2+l2pp);
                                        m2pp <= m2 - std::max(l2pp-l2, m2-l2pp); ++m2pp) {
                                    auto mbar1mbar2 = Mbar1[pack_lmlm(l1, m1, l1pp, m1pp)]
                                                    * Mbar2[pack_lmlm(l2, m2, l2pp, m2pp)];
                                    int m = m1pp + m2pp;
                                    int qmax = std::min(std::min(l1pp, l2pp),
                                                        (l1pp+l2pp-std::abs(m))/2);
                                    for (int q = 0; q <= qmax; ++q) {
                                        int l = l1pp + l2pp - 2*q;
                                        int ic = q*SZ2 + pack_lm(l, m);
                                        coef[ir*SZX+ic] += G[gind(l1pp, m1pp, l2pp, m2pp, q)] * mbar1mbar2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    dur[3] += iclock::now() - start;
}


void test_G() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 1.0);
    double rx = dis(gen);
    double ry = dis(gen);
    double rz = dis(gen);
    double rabs = std::sqrt(rx*rx + ry*ry + rz*rz);

    std::vector<double> Gaunt(SZ1*SZ1*(LMAX+1));
    gaunt_gen(Gaunt);

    for (int l1 = 0; l1 <= LMAX; ++l1) {
        for (int m1 = -l1; m1 <= l1; ++m1) {
            for (int l2 = 0; l2 <= LMAX; ++l2) {
                for (int m2 = -l2; m2 <= l2; ++m2) {
                    std::complex<double> val(0.0, 0.0);
                    int m = m1 + m2;
                    int qmax = std::min(std::min(l1, l2), (l1+l2-std::abs(m))/2);
                    for (int q = 0; q <= qmax; ++q) {
                        int l = l1 + l2 - 2*q;
                        if (std::abs(m) > l) {
                            break;
                        }
                        val += Gaunt[gind(l1, m1, l2, m2, q)] * std::pow(rabs, 2*q) * solid_harm(l, m, rx, ry, rz);
                    }
                    std::complex<double> ref = solid_harm(l1, m1, rx, ry, rz) * solid_harm(l2, m2, rx, ry, rz);
                    assert(std::abs(ref-val) < 1e-8);
                }
            }
        }
    }
}


void test_M() {
    std::vector<double> M(SZ1*SZ1);
    M_gen(M);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, 1);

    double r1x = dis(gen);
    double r1y = dis(gen);
    double r1z = dis(gen);
    double r2x = dis(gen);
    double r2y = dis(gen);
    double r2z = dis(gen);

    double rx = r1x + r2x;
    double ry = r1y + r2y;
    double rz = r1z + r2z;

    for (int l = 0; l <= LMAX; ++l) {
        for (int m = -l; m <= l; ++m) {
            std::complex<double> val(0.0, 0.0);

            for (int lp = 0; lp <= l; ++lp) {
                for (int mp = std::max(-lp, m+lp-l); mp <= std::min(lp, m+l-lp); ++mp) {
                    val += M[pack_lmlm(l, m, lp, mp)] * solid_harm(lp, mp, r1x, r1y, r1z) * solid_harm(l-lp, m-mp, r2x, r2y, r2z);        
                }
            }

            std::complex<double> ref = solid_harm(l, m, rx, ry, rz);
            assert(std::abs(ref-val) < 1e-8);
        }
    }
}


int main() {

    // Gaunt table (l1m1,l2m2,q) (with selection rule)
    std::vector<double> Gaunt(SZ1*SZ1*(LMAX+1));

    // translation table (lm,lpmp)
    std::vector<double> M(SZ1*SZ1);

    // M times the position-dependent part
    std::vector<std::complex<double>> mc0(SZ1*SZ1);

    // final expansion coefficient, (l1m1,l2m2,q,lm)
    std::vector<std::complex<double>> coef(SZ1*SZ1*SZX);

    // tabulate Gaunt & translation coefficient table
    gaunt_gen(Gaunt);
    test_G();

    M_gen(M);
    test_M();


    /*************************************************
     *      Product of two solid harmonics
     *************************************************/
    //------------ setup ------------
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, 1);

    // generate random positions r, A, B
    double rx = dis(gen);
    double ry = dis(gen);
    double rz = dis(gen);
    double Ax = dis(gen);
    double Ay = dis(gen);
    double Az = dis(gen);
    double Bx = dis(gen);
    double By = dis(gen);
    double Bz = dis(gen);

    // generate random exponents alpha & beta
    std::uniform_real_distribution<> dis2(0, 1);
    double alpha = dis2(gen);
    double beta = dis2(gen);

    //------------ compute expansion ------------
    // compute the position-dependent table
    int nt1 = 1000;
    auto start = iclock::now();
    for (int i = 0; i < nt1; ++i) {
        MC0(M, Ax-Bx, Ay-By, Az-Bz, mc0);
    }
    dur[0] = iclock::now() - start;
    printf("position part: %6.1f us (averaged by %i runs)\n",
            dur[0].count() * 1e6 / nt1, nt1);


    // compute the expansion coefficient
    start = iclock::now();
    int nt2 = 1000;
    for (int i = 0; i < nt2; ++i) {
        solid_harm_prod(Gaunt, mc0, alpha/beta, coef);
    }
    dur[1] = iclock::now() - start;
    printf("exponent part: %6.1f us (averaged by %i runs)\n",
            dur[1].count() * 1e6 / nt2, nt2);

    printf("- multiply   : %6.1f us\n", dur[2].count() * 1e6 / nt2);
    printf("- contraction: %6.1f us\n", dur[3].count() * 1e6 / nt2);

    //------------ verification ------------
    double Cx = (alpha*Ax + beta*Bx) / (alpha + beta);
    double Cy = (alpha*Ay + beta*By) / (alpha + beta);
    double Cz = (alpha*Az + beta*Bz) / (alpha + beta);

    double rAx = rx - Ax;
    double rAy = ry - Ay;
    double rAz = rz - Az;
    double rBx = rx - Bx;
    double rBy = ry - By;
    double rBz = rz - Bz;
    double rCx = rx - Cx;
    double rCy = ry - Cy;
    double rCz = rz - Cz;

    double rCabs = std::sqrt(rCx*rCx + rCy*rCy + rCz*rCz);

    auto unpack_qlm = [](int qlm, int& q, int& l, int& m) {
        q = qlm / SZ2;
        int lm = qlm % SZ2;
        l = std::sqrt(lm);
        m = lm - l*l - l;
    };

    for (int l1 = 0; l1 <= LMAX; ++l1) {
        for (int l2 = 0; l2 <= LMAX; ++l2) {
            for (int m1 = -l1; m1 <= l1; ++m1) {
                for (int m2 = -l2; m2 <= l2; ++m2) {
                    int ir = pack_lm(l1, m1) * SZ1 + pack_lm(l2, m2);

                    std::complex<double> val(0.0, 0.0);
                    for (int qlm = 0; qlm < SZX; ++qlm) {
                        int q, l, m;
                        unpack_qlm(qlm, q, l, m);
                        val += coef[ir*SZX + qlm] * std::pow(rCabs, 2*q)
                            * solid_harm(l, m, rCx, rCy, rCz);
                    }

                    std::complex<double> ref = solid_harm(l1, m1, rAx, rAy, rAz) * solid_harm(l2, m2, rBx, rBy, rBz);
                    assert(std::abs(ref-val) < 1e-8);
                }
            }
        }
    }

    return 0;
}





