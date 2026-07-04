#pragma once

#include <Eigen/Dense>
#include <libint2/cxxapi.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace miniqc {

inline std::size_t eri_index(std::size_t i,
                             std::size_t j,
                             std::size_t k,
                             std::size_t l,
                             std::size_t n2,
                             std::size_t n3,
                             std::size_t n4) {
    return ((i * n2 + j) * n3 + k) * n4 + l;
}

inline void check_density_matrix_shape(const libint2::BasisSet& basis,
                                       const Eigen::MatrixXd& D,
                                       const char* caller) {
    const int nbf = static_cast<int>(basis.nbf());
    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error(std::string(caller) + ": density matrix dimension mismatch");
    }
}

// Coulomb matrix only:
//   J_mn = sum_ls D_ls (mn|ls)
inline Eigen::MatrixXd build_j_direct(const libint2::BasisSet& basis,
                                      const Eigen::MatrixXd& D) {
    check_density_matrix_shape(basis, D, "build_j_direct");

    const int nbf = static_cast<int>(basis.nbf());
    const std::size_t nshell = basis.size();
    const auto& shell2bf = basis.shell2bf();
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nbf, nbf);

    const long long npairs =
        static_cast<long long>(nshell) * static_cast<long long>(nshell);

#pragma omp parallel
    {
        libint2::Engine engine(libint2::Operator::coulomb,
                               basis.max_nprim(),
                               basis.max_l());

#pragma omp for schedule(dynamic)
        for (long long pair12 = 0; pair12 < npairs; ++pair12) {
            const std::size_t s1 = static_cast<std::size_t>(pair12) / nshell;
            const std::size_t s2 = static_cast<std::size_t>(pair12) % nshell;

            const std::size_t bf1 = shell2bf[s1];
            const std::size_t bf2 = shell2bf[s2];
            const std::size_t n1 = basis[s1].size();
            const std::size_t n2 = basis[s2].size();

            for (std::size_t s3 = 0; s3 < nshell; ++s3) {
                for (std::size_t s4 = 0; s4 < nshell; ++s4) {
                    engine.compute2<libint2::Operator::coulomb,
                                    libint2::BraKet::xx_xx,
                                    0>(basis[s1], basis[s2], basis[s3], basis[s4]);

                    const double* buf = engine.results()[0];
                    if (buf == nullptr) continue;

                    const std::size_t bf3 = shell2bf[s3];
                    const std::size_t bf4 = shell2bf[s4];
                    const std::size_t n3 = basis[s3].size();
                    const std::size_t n4 = basis[s4].size();

                    for (std::size_t i = 0; i < n1; ++i) {
                        const int mu = static_cast<int>(bf1 + i);
                        for (std::size_t j = 0; j < n2; ++j) {
                            const int nu = static_cast<int>(bf2 + j);
                            double val = 0.0;

                            for (std::size_t k = 0; k < n3; ++k) {
                                const int lam = static_cast<int>(bf3 + k);
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const int sig = static_cast<int>(bf4 + l);
                                    const std::size_t idx = eri_index(i, j, k, l, n2, n3, n4);
                                    val += D(lam, sig) * buf[idx];
                                }
                            }

                            J(mu, nu) += val;
                        }
                    }
                }
            }
        }
    }

    return 0.5 * (J + J.transpose());
}

// Exact-exchange matrix:
//   K_mn = sum_ls D_ls (ml|ns)
//
// D is assumed to be the same spin-summed closed-shell density used elsewhere
// in this educational code.  Hybrid RKS will use this as
//   F = Hcore + J + Vxc - a_x K.
inline Eigen::MatrixXd build_k_direct(const libint2::BasisSet& basis,
                                      const Eigen::MatrixXd& D) {
    check_density_matrix_shape(basis, D, "build_k_direct");

    const int nbf = static_cast<int>(basis.nbf());
    const std::size_t nshell = basis.size();
    const auto& shell2bf = basis.shell2bf();
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(nbf, nbf);

    const long long npairs =
        static_cast<long long>(nshell) * static_cast<long long>(nshell);

#pragma omp parallel
    {
        libint2::Engine engine(libint2::Operator::coulomb,
                               basis.max_nprim(),
                               basis.max_l());

#pragma omp for schedule(dynamic)
        for (long long pair12 = 0; pair12 < npairs; ++pair12) {
            const std::size_t s1 = static_cast<std::size_t>(pair12) / nshell;
            const std::size_t s2 = static_cast<std::size_t>(pair12) % nshell;

            const std::size_t bf1 = shell2bf[s1];
            const std::size_t bf2 = shell2bf[s2];
            const std::size_t n1 = basis[s1].size();
            const std::size_t n2 = basis[s2].size();

            // For target K_{mu,nu}, compute quartets in order
            // (s1,s3|s2,s4), so the local integral index is (i,k,j,l).
            for (std::size_t s3 = 0; s3 < nshell; ++s3) {
                for (std::size_t s4 = 0; s4 < nshell; ++s4) {
                    engine.compute2<libint2::Operator::coulomb,
                                    libint2::BraKet::xx_xx,
                                    0>(basis[s1], basis[s3], basis[s2], basis[s4]);

                    const double* buf = engine.results()[0];
                    if (buf == nullptr) continue;

                    const std::size_t bf3 = shell2bf[s3];
                    const std::size_t bf4 = shell2bf[s4];
                    const std::size_t n3 = basis[s3].size();
                    const std::size_t n4 = basis[s4].size();

                    for (std::size_t i = 0; i < n1; ++i) {
                        const int mu = static_cast<int>(bf1 + i);
                        for (std::size_t j = 0; j < n2; ++j) {
                            const int nu = static_cast<int>(bf2 + j);
                            double val = 0.0;

                            for (std::size_t k = 0; k < n3; ++k) {
                                const int lam = static_cast<int>(bf3 + k);
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const int sig = static_cast<int>(bf4 + l);
                                    const double d_ls = D(lam, sig);
                                    if (std::abs(d_ls) < 1.0e-16) continue;

                                    const std::size_t idx = eri_index(i, k, j, l, n3, n2, n4);
                                    val += d_ls * buf[idx];
                                }
                            }

                            K(mu, nu) += val;
                        }
                    }
                }
            }
        }
    }

    return 0.5 * (K + K.transpose());
}

inline Eigen::MatrixXd build_rhf_g_from_jk(const Eigen::MatrixXd& J,
                                           const Eigen::MatrixXd& K) {
    if (J.rows() != K.rows() || J.cols() != K.cols()) {
        throw std::runtime_error("build_rhf_g_from_jk: J/K dimension mismatch");
    }
    return J - 0.5 * K;
}

inline Eigen::MatrixXd build_hybrid_g_from_jk(const Eigen::MatrixXd& J,
                                              const Eigen::MatrixXd& K,
                                              double exact_exchange_fraction) {
    if (J.rows() != K.rows() || J.cols() != K.cols()) {
        throw std::runtime_error("build_hybrid_g_from_jk: J/K dimension mismatch");
    }
    return J - exact_exchange_fraction * K;
}

// Direct closed-shell RHF two-electron Fock contribution:
//   G_mn = sum_ls D_ls [ (mn|ls) - 1/2 (ml|ns) ]
// D is the spin-summed closed-shell density: D_mn = 2 sum_i C_mi C_ni.
//
// This routine keeps the old single-pass RHF implementation for performance.
// The new build_j_direct/build_k_direct interface above is provided for hybrid
// DFT and for explicit J/K testing.
inline Eigen::MatrixXd build_g_direct(const libint2::BasisSet& basis,
                                      const Eigen::MatrixXd& D) {
    check_density_matrix_shape(basis, D, "build_g_direct");

    const int nbf = static_cast<int>(basis.nbf());
    const std::size_t nshell = basis.size();
    const auto& shell2bf = basis.shell2bf();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nbf, nbf);

    const long npairs = static_cast<long>(nshell * nshell);

#pragma omp parallel
    {
        libint2::Engine engJ(libint2::Operator::coulomb,
                             basis.max_nprim(),
                             basis.max_l());
        libint2::Engine engK(libint2::Operator::coulomb,
                             basis.max_nprim(),
                             basis.max_l());

#pragma omp for schedule(dynamic)
        for (long pair12 = 0; pair12 < npairs; ++pair12) {
            const std::size_t s1 = static_cast<std::size_t>(pair12) / nshell;
            const std::size_t s2 = static_cast<std::size_t>(pair12) % nshell;

            const std::size_t bf1 = shell2bf[s1];
            const std::size_t bf2 = shell2bf[s2];
            const std::size_t n1 = basis[s1].size();
            const std::size_t n2 = basis[s2].size();

            for (std::size_t s3 = 0; s3 < nshell; ++s3) {
                for (std::size_t s4 = 0; s4 < nshell; ++s4) {
                    const std::size_t bf3 = shell2bf[s3];
                    const std::size_t bf4 = shell2bf[s4];
                    const std::size_t n3 = basis[s3].size();
                    const std::size_t n4 = basis[s4].size();

                    engJ.compute2<libint2::Operator::coulomb,
                                  libint2::BraKet::xx_xx,
                                  0>(basis[s1], basis[s2], basis[s3], basis[s4]);
                    const double* bufJ = engJ.results()[0];

                    engK.compute2<libint2::Operator::coulomb,
                                  libint2::BraKet::xx_xx,
                                  0>(basis[s1], basis[s3], basis[s2], basis[s4]);
                    const double* bufK = engK.results()[0];

                    if (!bufJ && !bufK) continue;

                    for (std::size_t i = 0; i < n1; ++i) {
                        const int mu = static_cast<int>(bf1 + i);
                        for (std::size_t j = 0; j < n2; ++j) {
                            const int nu = static_cast<int>(bf2 + j);
                            double gij = 0.0;

                            for (std::size_t k = 0; k < n3; ++k) {
                                const int lam = static_cast<int>(bf3 + k);
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const int sig = static_cast<int>(bf4 + l);
                                    const double d_ls = D(lam, sig);
                                    if (std::abs(d_ls) < 1.0e-16) continue;

                                    const double J = bufJ ?
                                        bufJ[eri_index(i, j, k, l, n2, n3, n4)] : 0.0;

                                    // K quartet order is (s1,s3|s2,s4), so local order is (i,k,j,l).
                                    const double K = bufK ?
                                        bufK[eri_index(i, k, j, l, n3, n2, n4)] : 0.0;

                                    gij += d_ls * (J - 0.5 * K);
                                }
                            }

                            G(mu, nu) += gij;
                        }
                    }
                }
            }
        }
    }

    return 0.5 * (G + G.transpose());
}

}  // namespace miniqc
