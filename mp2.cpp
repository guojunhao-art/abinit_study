#include "mp2_v3_direct_t1.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace miniqc {

namespace {

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error(std::string("tensor dimension overflow while computing ") + label);
    }
    return a * b;
}

std::size_t checked_size4(std::size_t d0, std::size_t d1, std::size_t d2, std::size_t d3) {
    std::size_t x = checked_mul(d0, d1, "d0*d1");
    x = checked_mul(x, d2, "d0*d1*d2");
    x = checked_mul(x, d3, "d0*d1*d2*d3");
    return x;
}

inline double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    using seconds = std::chrono::duration<double>;
    return std::chrono::duration_cast<seconds>(std::chrono::steady_clock::now() - t0).count();
}

inline double tensor_mb(std::size_t n_double) {
    return static_cast<double>(n_double * sizeof(double)) / (1024.0 * 1024.0);
}

void decode3_shell(std::size_t idx,
                   std::size_t nshell,
                   std::size_t& s2, std::size_t& s3, std::size_t& s4) {
    s4 = idx % nshell; idx /= nshell;
    s3 = idx % nshell; idx /= nshell;
    s2 = idx;
}

void decode4_rect(std::size_t idx,
                  std::size_t d1, std::size_t d2, std::size_t d3,
                  std::size_t& a, std::size_t& b, std::size_t& c, std::size_t& d) {
    d = idx % d3; idx /= d3;
    c = idx % d2; idx /= d2;
    b = idx % d1; idx /= d1;
    a = idx;
}

void check_dimensions_for_mp2(std::size_t nbf, int nocc, const Eigen::MatrixXd& C) {
    if (nbf == 0) throw std::runtime_error("zero basis functions in MP2 V3");
    if (nocc <= 0 || nocc >= static_cast<int>(nbf)) {
        throw std::runtime_error("invalid nocc in MP2 V3");
    }
    if (static_cast<std::size_t>(C.rows()) != nbf || static_cast<std::size_t>(C.cols()) != nbf) {
        throw std::runtime_error("MO coefficient matrix C must be nbf x nbf in MP2 V3");
    }
    if (!C.allFinite()) throw std::runtime_error("MO coefficient matrix C contains non-finite values");
}

void check_memory_guard(std::size_t o, std::size_t v, std::size_t n, const MP2Options& options) {
    const double t1_mb = tensor_mb(checked_size4(o, n, n, n));
    const double t2_mb = tensor_mb(checked_size4(o, v, n, n));
    const double t3_mb = tensor_mb(checked_size4(o, v, o, n));
    const double ovov_mb = tensor_mb(checked_size4(o, v, o, v));
    const double max_mb = std::max(std::max(t1_mb, t2_mb), std::max(t3_mb, ovov_mb));

    if (options.max_intermediate_mb > 0.0 && max_mb > options.max_intermediate_mb) {
        throw std::runtime_error(
            "MP2 V3 refused this calculation: estimated largest intermediate = " +
            std::to_string(max_mb) + " MB, max_intermediate_mb = " +
            std::to_string(options.max_intermediate_mb) +
            ". Increase MP2Options::max_intermediate_mb only if you have enough memory.");
    }
}

}  // namespace

Tensor4DRect::Tensor4DRect(std::size_t d0, std::size_t d1, std::size_t d2, std::size_t d3)
    : d0_(d0), d1_(d1), d2_(d2), d3_(d3), data_(checked_size4(d0, d1, d2, d3), 0.0) {}

Tensor4DRect build_t1_direct_from_libint(const libint2::BasisSet& basis,
                                         const Eigen::MatrixXd& C,
                                         int nocc,
                                         const MP2Options& options) {
    const std::size_t n = basis.nbf();
    const std::size_t o = static_cast<std::size_t>(nocc);
    const std::size_t nshell = basis.size();

    check_dimensions_for_mp2(n, nocc, C);

    if (options.verbose) {
        std::cout << "MP2 V3: direct first-index transformation from Libint shell quartets\n";
        std::cout << "        no dense AO ERI tensor will be stored\n";
        std::cout << "        t1(o,N,N,N) memory = " << std::fixed << std::setprecision(2)
                  << tensor_mb(checked_size4(o, n, n, n)) << " MB\n";
    }

    auto t0 = std::chrono::steady_clock::now();
    Tensor4DRect t1(o, n, n, n);
    const auto& shell2bf = basis.shell2bf();
    const std::size_t ntasks = nshell * nshell * nshell;  // (snu, slam, ssig)

#pragma omp parallel
    {
        libint2::Engine engine(libint2::Operator::coulomb, basis.max_nprim(), basis.max_l());

#pragma omp for schedule(dynamic)
        for (long long task_ll = 0; task_ll < static_cast<long long>(ntasks); ++task_ll) {
            std::size_t snu, slam, ssig;
            decode3_shell(static_cast<std::size_t>(task_ll), nshell, snu, slam, ssig);

            // This task owns t1(:, nu-block, lam-block, sig-block). The summation
            // over smu is performed inside this task, so no two threads write to
            // the same t1 elements.
            const std::size_t bf_nu = shell2bf[snu];
            const std::size_t bf_lam = shell2bf[slam];
            const std::size_t bf_sig = shell2bf[ssig];
            const std::size_t nnu = basis[snu].size();
            const std::size_t nlam = basis[slam].size();
            const std::size_t nsig = basis[ssig].size();

            for (std::size_t smu = 0; smu < nshell; ++smu) {
                const std::size_t bf_mu = shell2bf[smu];
                const std::size_t nmu = basis[smu].size();

                engine.compute2<libint2::Operator::coulomb, libint2::BraKet::xx_xx, 0>(
                    basis[smu], basis[snu], basis[slam], basis[ssig]);
                const double* buf = engine.results()[0];
                if (!buf) continue;

                for (std::size_t u = 0; u < nmu; ++u) {
                    const std::size_t mu = bf_mu + u;

                    for (std::size_t i = 0; i < o; ++i) {
                        const double c_mu_i = C(static_cast<int>(mu), static_cast<int>(i));
                        if (std::abs(c_mu_i) < 1.0e-16) continue;

                        for (std::size_t v = 0; v < nnu; ++v) {
                            const std::size_t nu = bf_nu + v;
                            for (std::size_t w = 0; w < nlam; ++w) {
                                const std::size_t lam = bf_lam + w;
                                const std::size_t base = ((u * nnu + v) * nlam + w) * nsig;
                                for (std::size_t x = 0; x < nsig; ++x) {
                                    const std::size_t sig = bf_sig + x;
                                    t1(i, nu, lam, sig) += c_mu_i * buf[base + x];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (options.verbose) {
        std::cout << "MP2 V3: direct t1 build finished in "
                  << std::fixed << std::setprecision(3) << seconds_since(t0) << " s\n";
    }
    return t1;
}

Tensor4DRect transform_t1_to_ovov(Tensor4DRect t1,
                                  const Eigen::MatrixXd& C,
                                  int nocc,
                                  const MP2Options& options) {
    const std::size_t o = t1.dim0();
    const std::size_t n = t1.dim1();
    if (t1.dim2() != n || t1.dim3() != n) {
        throw std::runtime_error("t1 must have dimensions nocc x nbf x nbf x nbf in MP2 V3");
    }
    check_dimensions_for_mp2(n, nocc, C);
    if (o != static_cast<std::size_t>(nocc)) {
        throw std::runtime_error("t1 dim0 does not match nocc in MP2 V3");
    }

    const std::size_t v = n - o;
    check_memory_guard(o, v, n, options);
    auto t0 = std::chrono::steady_clock::now();

    if (options.verbose) {
        std::cout << "MP2 V3: completing t1 -> OVOV transformation\n";
        std::cout << "        t2(o,v,N,N)    memory = " << std::fixed << std::setprecision(2)
                  << tensor_mb(checked_size4(o, v, n, n)) << " MB\n";
        std::cout << "        t3(o,v,o,N)    memory = "
                  << tensor_mb(checked_size4(o, v, o, n)) << " MB\n";
        std::cout << "        ovov(o,v,o,v)  memory = "
                  << tensor_mb(checked_size4(o, v, o, v)) << " MB\n";
    }

    // t2(i,a,lam,sig) = sum_nu C(nu,A) t1(i,nu,lam,sig), A = nocc + a
    Tensor4DRect t2(o, v, n, n);
    const std::size_t total_t2 = t2.size();
#pragma omp parallel for schedule(static)
    for (long long idx_ll = 0; idx_ll < static_cast<long long>(total_t2); ++idx_ll) {
        const std::size_t idx = static_cast<std::size_t>(idx_ll);
        std::size_t i, a, lam, sig;
        decode4_rect(idx, v, n, n, i, a, lam, sig);
        const std::size_t A = o + a;
        double sum = 0.0;
        for (std::size_t nu = 0; nu < n; ++nu) {
            sum += C(static_cast<int>(nu), static_cast<int>(A)) * t1(i, nu, lam, sig);
        }
        t2(i, a, lam, sig) = sum;
    }
    t1 = Tensor4DRect{};

    // t3(i,a,j,sig) = sum_lam C(lam,j) t2(i,a,lam,sig)
    Tensor4DRect t3(o, v, o, n);
    const std::size_t total_t3 = t3.size();
#pragma omp parallel for schedule(static)
    for (long long idx_ll = 0; idx_ll < static_cast<long long>(total_t3); ++idx_ll) {
        const std::size_t idx = static_cast<std::size_t>(idx_ll);
        std::size_t i, a, j, sig;
        decode4_rect(idx, v, o, n, i, a, j, sig);
        double sum = 0.0;
        for (std::size_t lam = 0; lam < n; ++lam) {
            sum += C(static_cast<int>(lam), static_cast<int>(j)) * t2(i, a, lam, sig);
        }
        t3(i, a, j, sig) = sum;
    }
    t2 = Tensor4DRect{};

    // ovov(i,a,j,b) = sum_sig C(sig,B) t3(i,a,j,sig), B = nocc + b
    Tensor4DRect ovov(o, v, o, v);
    const std::size_t total_ovov = ovov.size();
#pragma omp parallel for schedule(static)
    for (long long idx_ll = 0; idx_ll < static_cast<long long>(total_ovov); ++idx_ll) {
        const std::size_t idx = static_cast<std::size_t>(idx_ll);
        std::size_t i, a, j, b;
        decode4_rect(idx, v, o, v, i, a, j, b);
        const std::size_t B = o + b;
        double sum = 0.0;
        for (std::size_t sig = 0; sig < n; ++sig) {
            sum += C(static_cast<int>(sig), static_cast<int>(B)) * t3(i, a, j, sig);
        }
        ovov(i, a, j, b) = sum;
    }

    if (options.verbose) {
        std::cout << "MP2 V3: t1 -> OVOV transformation finished in "
                  << std::fixed << std::setprecision(3) << seconds_since(t0) << " s\n";
    }
    return ovov;
}

MP2Result compute_rhf_mp2_energy(const libint2::BasisSet& basis,
                                 const Eigen::MatrixXd& C,
                                 const Eigen::VectorXd& eps,
                                 int nocc,
                                 double hf_total_energy,
                                 const MP2Options& options) {
    const std::size_t nbf = basis.nbf();
    check_dimensions_for_mp2(nbf, nocc, C);
    if (static_cast<std::size_t>(eps.size()) != nbf) {
        throw std::runtime_error("orbital-energy vector has wrong length for MP2 V3");
    }

    const std::size_t o = static_cast<std::size_t>(nocc);
    const std::size_t v = nbf - o;
    check_memory_guard(o, v, nbf, options);

    Tensor4DRect t1 = build_t1_direct_from_libint(basis, C, nocc, options);
    const Tensor4DRect ovov = transform_t1_to_ovov(std::move(t1), C, nocc, options);

    auto t0 = std::chrono::steady_clock::now();
    double e_corr = 0.0;

#pragma omp parallel for reduction(+:e_corr) collapse(2) schedule(static)
    for (long long i_ll = 0; i_ll < static_cast<long long>(o); ++i_ll) {
        for (long long j_ll = 0; j_ll < static_cast<long long>(o); ++j_ll) {
            const std::size_t i = static_cast<std::size_t>(i_ll);
            const std::size_t j = static_cast<std::size_t>(j_ll);
            for (std::size_t a = 0; a < v; ++a) {
                const std::size_t A = o + a;
                for (std::size_t b = 0; b < v; ++b) {
                    const std::size_t B = o + b;
                    const double iajb = ovov(i, a, j, b);  // (i a | j b)
                    const double ibja = ovov(i, b, j, a);  // (i b | j a)
                    const double denom = eps(static_cast<int>(i)) + eps(static_cast<int>(j))
                                       - eps(static_cast<int>(A)) - eps(static_cast<int>(B));
                    e_corr += iajb * (2.0 * iajb - ibja) / denom;
                }
            }
        }
    }

    MP2Result result;
    result.correlation_energy = e_corr;
    result.total_energy = hf_total_energy + e_corr;
    result.nbf = nbf;
    result.nocc = o;
    result.nvir = v;

    if (options.verbose) {
        std::cout << "MP2 V3: energy summation finished in "
                  << std::fixed << std::setprecision(3) << seconds_since(t0) << " s\n";
        std::cout << "MP2 V3: nocc = " << result.nocc << ", nvir = " << result.nvir << "\n";
    }

    return result;
}

}  // namespace miniqc
