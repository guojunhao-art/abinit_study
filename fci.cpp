#include "fci.hpp"
#include "cisd.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace miniqc {

namespace {

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error(std::string("integer overflow while computing ") + label);
    }
    return a * b;
}

std::size_t checked_size4(std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
    std::size_t x = checked_mul(a, b, "d0*d1");
    x = checked_mul(x, c, "d0*d1*d2");
    x = checked_mul(x, d, "d0*d1*d2*d3");
    return x;
}

inline double tensor_mb(std::size_t n_double) {
    return static_cast<double>(n_double * sizeof(double)) / (1024.0 * 1024.0);
}

inline double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    using seconds = std::chrono::duration<double>;
    return std::chrono::duration_cast<seconds>(std::chrono::steady_clock::now() - t0).count();
}

void decode4(std::size_t idx, std::size_t n,
             std::size_t& a, std::size_t& b, std::size_t& c, std::size_t& d) {
    d = idx % n; idx /= n;
    c = idx % n; idx /= n;
    b = idx % n; idx /= n;
    a = idx;
}

inline std::size_t eri_buf_index(std::size_t i, std::size_t j,
                                 std::size_t k, std::size_t l,
                                 std::size_t n2, std::size_t n3, std::size_t n4) {
    return ((i * n2 + j) * n3 + k) * n4 + l;
}

std::vector<std::uint64_t> generate_bitstrings(int norb, int nelec) {
    if (norb < 0 || norb > 63) throw std::runtime_error("FCI bitstring generator requires norb <= 63");
    if (nelec < 0 || nelec > norb) throw std::runtime_error("invalid electron count for bitstring generation");

    std::vector<std::uint64_t> strings;

    std::function<void(int, int, std::uint64_t)> rec =
        [&](int start, int left, std::uint64_t bits) {
            if (left == 0) {
                strings.push_back(bits);
                return;
            }
            for (int p = start; p <= norb - left; ++p) {
                rec(p + 1, left - 1, bits | (std::uint64_t{1} << p));
            }
        };

    rec(0, nelec, 0ULL);
    return strings;
}

std::vector<std::uint64_t> generate_fixed_alpha_beta_dets(int nmo, int nalpha, int nbeta) {
    if (nmo <= 0 || nmo > 31) {
        // We use one uint64_t for alpha|beta, with beta bits shifted by nmo.
        throw std::runtime_error("this teaching FCI implementation currently requires nmo <= 31");
    }

    const auto alpha_strings = generate_bitstrings(nmo, nalpha);
    const auto beta_strings = generate_bitstrings(nmo, nbeta);

    std::vector<std::uint64_t> dets;
    dets.reserve(alpha_strings.size() * beta_strings.size());
    for (std::uint64_t a : alpha_strings) {
        for (std::uint64_t b : beta_strings) {
            dets.push_back(a | (b << nmo));
        }
    }
    return dets;
}

inline int popcount64(std::uint64_t x) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) {
        x &= (x - 1);
        ++count;
    }
    return count;
#endif
}

struct OpResult {
    bool ok = false;
    std::uint64_t det = 0;
    int sign = 1;
};

OpResult annihilate(std::uint64_t det, int orb) {
    const std::uint64_t mask = std::uint64_t{1} << orb;
    if ((det & mask) == 0) return {false, det, 0};

    const std::uint64_t below = mask - 1;
    const int nbelow = popcount64(det & below);
    const int sign = (nbelow % 2 == 0) ? 1 : -1;
    return {true, det ^ mask, sign};
}

OpResult create(std::uint64_t det, int orb) {
    const std::uint64_t mask = std::uint64_t{1} << orb;
    if ((det & mask) != 0) return {false, det, 0};

    const std::uint64_t below = mask - 1;
    const int nbelow = popcount64(det & below);
    const int sign = (nbelow % 2 == 0) ? 1 : -1;
    return {true, det | mask, sign};
}

inline int spatial_index(int spinorb, int nmo) {
    return spinorb < nmo ? spinorb : spinorb - nmo;
}

inline int spin_index(int spinorb, int nmo) {
    return spinorb < nmo ? 0 : 1;  // 0 = alpha, 1 = beta
}

inline double h_spin(const Eigen::MatrixXd& h_mo, int P, int Q, int nmo) {
    if (spin_index(P, nmo) != spin_index(Q, nmo)) return 0.0;
    return h_mo(spatial_index(P, nmo), spatial_index(Q, nmo));
}

// <PQ||RS> in spin-orbital notation.
// eri_mo uses chemists' spatial notation (pq|rs) = int p(1)q(1) r12^-1 r(2)s(2).
// Physicists' <PQ|RS> = delta_PR delta_QS * (p r | q s).
inline double antisym_spin_eri(const Tensor4D& eri_mo, int P, int Q, int R, int S, int nmo) {
    const int p = spatial_index(P, nmo);
    const int q = spatial_index(Q, nmo);
    const int r = spatial_index(R, nmo);
    const int s = spatial_index(S, nmo);

    const int spP = spin_index(P, nmo);
    const int spQ = spin_index(Q, nmo);
    const int spR = spin_index(R, nmo);
    const int spS = spin_index(S, nmo);

    double value = 0.0;
    if (spP == spR && spQ == spS) value += eri_mo(p, r, q, s);
    if (spP == spS && spQ == spR) value -= eri_mo(p, s, q, r);
    return value;
}

void print_largest_ci_coefficients(const Eigen::VectorXd& coeff,
                                   std::vector<std::uint64_t>& dets,
                                   int nmo,
                                   std::size_t nprint) {
    std::vector<std::pair<double, std::size_t>> order;
    order.reserve(static_cast<std::size_t>(coeff.size()));
    for (int i = 0; i < coeff.size(); ++i) {
        order.push_back({std::abs(coeff(i)), static_cast<std::size_t>(i)});
    }
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    std::cout << "Largest CI coefficients:\n";
    for (std::size_t k = 0; k < std::min(nprint, order.size()); ++k) {
        const std::size_t idx = order[k].second;
        const std::uint64_t det = dets[idx];
        const std::uint64_t alpha = det & ((std::uint64_t{1} << nmo) - 1);
        const std::uint64_t beta = det >> nmo;
        std::cout << "  " << std::setw(6) << idx
                  << "  c = " << std::setw(18) << std::setprecision(12) << coeff(static_cast<int>(idx))
                  << "  alpha_bits = 0x" << std::hex << alpha
                  << "  beta_bits = 0x" << beta << std::dec << "\n";
    }
}

}  // namespace

Tensor4D::Tensor4D(std::size_t n0, std::size_t n1, std::size_t n2, std::size_t n3)
    : n0_(n0), n1_(n1), n2_(n2), n3_(n3), data_(checked_size4(n0, n1, n2, n3), 0.0) {}

Tensor4D build_full_mo_eri_from_libint(const libint2::BasisSet& basis,
                                       const Eigen::MatrixXd& C,
                                       const FCIOptions& options) {
    const std::size_t n = basis.nbf();
    if (n == 0) throw std::runtime_error("FCI/CISD/CID: zero basis functions");
    if (n > options.max_nmo) {
        throw std::runtime_error("FCI refused this calculation: nmo = " + std::to_string(n) +
                                 ", max_nmo = " + std::to_string(options.max_nmo) +
                                 ". Use a smaller basis or increase FCIOptions::max_nmo knowingly.");
    }
    if (static_cast<std::size_t>(C.rows()) != n || static_cast<std::size_t>(C.cols()) != n) {
        throw std::runtime_error("FCI/CISD/CID: C must be nbf x nbf");
    }

    const double one_tensor_mb = tensor_mb(checked_size4(n, n, n, n));
    if (options.max_tensor_mb > 0.0 && one_tensor_mb > options.max_tensor_mb) {
        throw std::runtime_error("FCI refused this calculation: one N^4 tensor = " +
                                 std::to_string(one_tensor_mb) + " MB, max_tensor_mb = " +
                                 std::to_string(options.max_tensor_mb));
    }

    if (options.verbose) {
        std::cout << "FCI/CISD/CID: building full MO ERI tensor, nmo = " << n
                  << ", memory per N^4 tensor = " << std::fixed << std::setprecision(2)
                  << one_tensor_mb << " MB\n";
    }

    auto t0 = std::chrono::steady_clock::now();
    Tensor4D ao(n, n, n, n);
    const auto& shell2bf = basis.shell2bf();
    const std::size_t nshell = basis.size();

    libint2::Engine engine(libint2::Operator::coulomb, basis.max_nprim(), basis.max_l());
    for (std::size_t s1 = 0; s1 < nshell; ++s1) {
        for (std::size_t s2 = 0; s2 < nshell; ++s2) {
            for (std::size_t s3 = 0; s3 < nshell; ++s3) {
                for (std::size_t s4 = 0; s4 < nshell; ++s4) {
                    engine.compute2<libint2::Operator::coulomb, libint2::BraKet::xx_xx, 0>(
                        basis[s1], basis[s2], basis[s3], basis[s4]);
                    const double* buf = engine.results()[0];
                    if (!buf) continue;

                    const std::size_t bf1 = shell2bf[s1];
                    const std::size_t bf2 = shell2bf[s2];
                    const std::size_t bf3 = shell2bf[s3];
                    const std::size_t bf4 = shell2bf[s4];
                    const std::size_t n1 = basis[s1].size();
                    const std::size_t n2 = basis[s2].size();
                    const std::size_t n3 = basis[s3].size();
                    const std::size_t n4 = basis[s4].size();

                    for (std::size_t i = 0; i < n1; ++i) {
                        for (std::size_t j = 0; j < n2; ++j) {
                            for (std::size_t k = 0; k < n3; ++k) {
                                const std::size_t base = ((i * n2 + j) * n3 + k) * n4;
                                for (std::size_t l = 0; l < n4; ++l) {
                                    ao(bf1 + i, bf2 + j, bf3 + k, bf4 + l) = buf[base + l];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (options.verbose) {
        std::cout << "FCI/CISD/CID: AO ERI tensor built in " << std::fixed << std::setprecision(3)
                  << seconds_since(t0) << " s\n";
        std::cout << "FCI/CISD/CID: transforming AO ERI to full MO ERI\n";
    }

    auto t1_start = std::chrono::steady_clock::now();
    Tensor4D t1(n, n, n, n);
    const std::size_t total = checked_size4(n, n, n, n);

#pragma omp parallel for schedule(static)
    for (long long idx_ll = 0; idx_ll < static_cast<long long>(total); ++idx_ll) {
        std::size_t p, nu, lam, sig;
        decode4(static_cast<std::size_t>(idx_ll), n, p, nu, lam, sig);
        double sum = 0.0;
        for (std::size_t mu = 0; mu < n; ++mu) {
            sum += C(static_cast<int>(mu), static_cast<int>(p)) * ao(mu, nu, lam, sig);
        }
        t1(p, nu, lam, sig) = sum;
    }
    ao = Tensor4D{};  // release memory before allocating later large intermediates

    Tensor4D t2(n, n, n, n);
#pragma omp parallel for schedule(static)
    for (long long idx_ll = 0; idx_ll < static_cast<long long>(total); ++idx_ll) {
        std::size_t p, q, lam, sig;
        decode4(static_cast<std::size_t>(idx_ll), n, p, q, lam, sig);
        double sum = 0.0;
        for (std::size_t nu = 0; nu < n; ++nu) {
            sum += C(static_cast<int>(nu), static_cast<int>(q)) * t1(p, nu, lam, sig);
        }
        t2(p, q, lam, sig) = sum;
    }
    t1 = Tensor4D{};

    Tensor4D t3(n, n, n, n);
#pragma omp parallel for schedule(static)
    for (long long idx_ll = 0; idx_ll < static_cast<long long>(total); ++idx_ll) {
        std::size_t p, q, r, sig;
        decode4(static_cast<std::size_t>(idx_ll), n, p, q, r, sig);
        double sum = 0.0;
        for (std::size_t lam = 0; lam < n; ++lam) {
            sum += C(static_cast<int>(lam), static_cast<int>(r)) * t2(p, q, lam, sig);
        }
        t3(p, q, r, sig) = sum;
    }
    t2 = Tensor4D{};

    Tensor4D mo(n, n, n, n);
#pragma omp parallel for schedule(static)
    for (long long idx_ll = 0; idx_ll < static_cast<long long>(total); ++idx_ll) {
        std::size_t p, q, r, s;
        decode4(static_cast<std::size_t>(idx_ll), n, p, q, r, s);
        double sum = 0.0;
        for (std::size_t sig = 0; sig < n; ++sig) {
            sum += C(static_cast<int>(sig), static_cast<int>(s)) * t3(p, q, r, sig);
        }
        mo(p, q, r, s) = sum;
    }

    if (options.verbose) {
        std::cout << "FCI/CISD/CID: full AO -> MO transformation finished in "
                  << std::fixed << std::setprecision(3) << seconds_since(t1_start) << " s\n";
    }
    return mo;
}

FCIResult compute_fci_energy(const libint2::BasisSet& basis,
                             const Eigen::MatrixXd& Hcore,
                             const Eigen::MatrixXd& C,
                             int nelec,
                             double nuclear_repulsion,
                             const FCIOptions& options,
                             int func) {
    const int nmo = static_cast<int>(basis.nbf());
    if (nelec <= 0 || nelec > 2 * nmo) throw std::runtime_error("FCI/CISD/CID: invalid electron count");
    if (nelec % 2 != 0) throw std::runtime_error("this FCI/CISD/CID driver currently assumes closed-shell Nalpha=Nbeta");

    const int nalpha = nelec / 2;
    const int nbeta = nelec / 2;
    const int nspin = 2 * nmo;
    if (nspin > 62) throw std::runtime_error("FCI/CISD/CID: too many spin orbitals for uint64_t bitstrings");

    const Eigen::MatrixXd h_mo = C.transpose() * Hcore * C;
    Tensor4D eri_mo = build_full_mo_eri_from_libint(basis, C, options);

    std::vector<std::uint64_t> dets; 
    if (func == 0) dets = generate_fixed_alpha_beta_dets(nmo, nalpha, nbeta);//FCI
    if (func == 1) dets = generate_cisd_determinants(nmo, nalpha, nbeta, true);//CISD
    if (func == 2) dets = generate_cisd_determinants(nmo, nalpha, nbeta, false);//CID
    const std::size_t ndet = dets.size();
    if (ndet == 0) throw std::runtime_error("FCI/CISD/CID: empty determinant space");
    if (ndet > options.max_determinants) {
        throw std::runtime_error("FCI/CISD/CID refused this calculation: ndet = " + std::to_string(ndet) +
                                 ", max_determinants = " + std::to_string(options.max_determinants));
    }
    std::string method;
    if (func == 0) method = "FCI";
    if (func == 1) method = "CISD";
    if (func == 2) method = "CID";

    if (options.verbose) {
        
        std::cout << method <<": determinant space with fixed Nalpha/Nbeta\n";
        std::cout << "     nmo   = " << nmo << "\n";
        std::cout << "     nelec = " << nelec << "\n";
        std::cout << "     nalpha= " << nalpha << ", nbeta = " << nbeta << "\n";
        std::cout << "     ndet  = " << ndet << "\n";
        std::cout << method <<": building dense Hamiltonian, memory = "
                  << std::fixed << std::setprecision(2)
                  << tensor_mb(ndet * ndet) << " MB\n";
    }

    std::unordered_map<std::uint64_t, int> det_to_index;
    det_to_index.reserve(ndet * 2);
    for (std::size_t i = 0; i < ndet; ++i) {
        det_to_index[dets[i]] = static_cast<int>(i);
    }

    auto t0 = std::chrono::steady_clock::now();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(static_cast<int>(ndet), static_cast<int>(ndet));

    // Integral-driven application of the second-quantized Hamiltonian to each ket determinant.
    for (std::size_t col = 0; col < ndet; ++col) {
        const std::uint64_t ket = dets[col];

        // One-electron part: sum_PQ h_PQ a_P^+ a_Q
        for (int Q = 0; Q < nspin; ++Q) {
            const auto aQ = annihilate(ket, Q);
            if (!aQ.ok) continue;

            for (int P = 0; P < nspin; ++P) {
                const double hpq = h_spin(h_mo, P, Q, nmo);
                if (std::abs(hpq) < 1.0e-14) continue;

                const auto cP = create(aQ.det, P);
                if (!cP.ok) continue;

                const auto it = det_to_index.find(cP.det);
                if (it != det_to_index.end()) {
                    H(it->second, static_cast<int>(col)) += static_cast<double>(aQ.sign * cP.sign) * hpq;
                }
            }
        }

        // Two-electron part: 1/4 sum_PQRS <PQ||RS> a_P^+ a_Q^+ a_S a_R
        for (int R = 0; R < nspin; ++R) {
            const auto aR = annihilate(ket, R);
            if (!aR.ok) continue;
            for (int S = 0; S < nspin; ++S) {
                const auto aS = annihilate(aR.det, S);
                if (!aS.ok) continue;
                const int sign_ann = aR.sign * aS.sign;

                for (int Q = 0; Q < nspin; ++Q) {
                    const auto cQ = create(aS.det, Q);
                    if (!cQ.ok) continue;
                    for (int P = 0; P < nspin; ++P) {
                        const double g = antisym_spin_eri(eri_mo, P, Q, R, S, nmo);
                        if (std::abs(g) < 1.0e-14) continue;

                        const auto cP = create(cQ.det, P);
                        if (!cP.ok) continue;

                        const auto it = det_to_index.find(cP.det);
                        if (it != det_to_index.end()) {
                            const int sign = sign_ann * cQ.sign * cP.sign;
                            H(it->second, static_cast<int>(col)) += 0.25 * static_cast<double>(sign) * g;
                        }
                    }
                }
            }
        }
    }

    H = 0.5 * (H + H.transpose());

    if (options.verbose) {
        std::cout << method <<": Hamiltonian built in " << std::fixed << std::setprecision(3)
                  << seconds_since(t0) << " s\n";
        std::cout << method <<": diagonalizing dense Hamiltonian\n";
    }

    auto td = std::chrono::steady_clock::now();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    if (es.info() != Eigen::Success) throw std::runtime_error("FCI/CISD/CID Hamiltonian diagonalization failed");

    FCIResult result;
    result.electronic_energy = es.eigenvalues()(0);
    result.total_energy = result.electronic_energy + nuclear_repulsion;
    result.nmo = static_cast<std::size_t>(nmo);
    result.nelec = nelec;
    result.nalpha = nalpha;
    result.nbeta = nbeta;
    result.ndet = ndet;
    result.ci_vector = es.eigenvectors().col(0);

    if (options.verbose) {
        std::cout << method <<": diagonalization finished in " << std::fixed << std::setprecision(3)
                  << seconds_since(td) << " s\n";
    }
    if (options.print_largest_coefficients) {
        print_largest_ci_coefficients(result.ci_vector, dets, nmo, options.n_coefficients_to_print);
    }

    return result;
}

}  // namespace miniqc
