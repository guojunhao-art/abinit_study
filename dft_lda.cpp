#include "dft_lda.hpp"
#include "dft_grid.hpp"


#include <xc.h>
#include <xc_funcs.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dft {

constexpr double pi = 3.141592653589793238462643383279502884;

class LibxcFunctionalHandle {
public:
    LibxcFunctionalHandle(int functional_id, int polarization) {
        const int status = xc_func_init(&func_, functional_id, polarization);
        if (status != 0) {
            std::ostringstream oss;
            oss << "Libxc xc_func_init failed for functional id "
                << functional_id << ", status = " << status;
            throw std::runtime_error(oss.str());
        }
        initialized_ = true;
    }

    LibxcFunctionalHandle(const LibxcFunctionalHandle&) = delete;
    LibxcFunctionalHandle& operator=(const LibxcFunctionalHandle&) = delete;

    ~LibxcFunctionalHandle() {
        if (initialized_) {
            xc_func_end(&func_);
        }
    }

    xc_func_type* get() { return &func_; }
    const xc_func_type* get() const { return &func_; }

private:
    xc_func_type func_{};
    bool initialized_ = false;
};

std::string lda_functional_name(LDAFunctional functional) {
    switch (functional) {
        case LDAFunctional::SlaterX:
            return "hand-written Slater X";
        case LDAFunctional::Libxc_LDA_X:
            return "Libxc LDA_X";
        case LDAFunctional::Libxc_LDA_X_PZ81C:
            return "Libxc LDA_X + LDA_C_PZ";
        case LDAFunctional::Libxc_GGA_PBE:
            return "Libxc PBE GGA: GGA_X_PBE + GGA_C_PBE";
        default:
            return "unknown XC functional";
    }
}

Eigen::Vector3d shell_origin(const libint2::Shell& sh) {
    return Eigen::Vector3d{sh.O[0], sh.O[1], sh.O[2]};
}

Eigen::MatrixXd symmetric_orthogonalization(const Eigen::MatrixXd& S,
                                            double threshold = 1.0e-10) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("symmetric_orthogonalization: eigensolver failed");
    }

    const auto& evals = es.eigenvalues();
    const auto& U = es.eigenvectors();

    Eigen::VectorXd inv_sqrt(evals.size());
    for (int i = 0; i < evals.size(); ++i) {
        if (evals(i) < threshold) {
            std::ostringstream oss;
            oss << "symmetric_orthogonalization: near-linear dependence, S eigenvalue = "
                << evals(i);
            throw std::runtime_error(oss.str());
        }
        inv_sqrt(i) = 1.0 / std::sqrt(evals(i));
    }

    return U * inv_sqrt.asDiagonal() * U.transpose();
}

Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int nocc) {
    const int nbf = static_cast<int>(C.rows());
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(nbf, nbf);

    for (int mu = 0; mu < nbf; ++mu) {
        for (int nu = 0; nu < nbf; ++nu) {
            double x = 0.0;
            for (int i = 0; i < nocc; ++i) {
                x += C(mu, i) * C(nu, i);
            }
            D(mu, nu) = 2.0 * x;
        }
    }

    return 0.5 * (D + D.transpose());
}

void diagonalize_fock(const Eigen::MatrixXd& F,
                      const Eigen::MatrixXd& X,
                      Eigen::VectorXd& eps,
                      Eigen::MatrixXd& C) {
    const Eigen::MatrixXd Fp = X.transpose() * F * X;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Fp);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("diagonalize_fock: eigensolver failed");
    }

    eps = es.eigenvalues();
    C = X * es.eigenvectors();
}

int ncartesian(int l) {
    return (l + 1) * (l + 2) / 2;
}

double int_power(double x, int n) {
    double y = 1.0;
    for (int i = 0; i < n; ++i) y *= x;
    return y;
}

double d_int_power(double x, int n) {
    if (n == 0) return 0.0;
    return static_cast<double>(n) * int_power(x, n - 1);
}

void add_shell_values_cartesian(const libint2::Shell& sh,
                                const Eigen::Vector3d& r,
                                Eigen::VectorXd& phi,
                                std::size_t bf0) {
    const Eigen::Vector3d A = shell_origin(sh);
    const double x = r(0) - A(0);
    const double y = r(1) - A(1);
    const double z = r(2) - A(2);
    const double r2 = x*x + y*y + z*z;

    std::size_t local = 0;

    for (const auto& contr : sh.contr) {
        const int l = contr.l;
        const int ncart = ncartesian(l);

        if (contr.pure) {
            std::ostringstream oss;
            oss << "eval_ao_values: pure/spherical shell detected with l=" << l
                << ". The current DFT grid AO evaluator supports Cartesian "
                << "Gaussian shells only. Use a Cartesian basis/shell setting, "
                << "or add a Libint-compatible Cartesian-to-spherical transform.";
            throw std::runtime_error(oss.str());
        }

        if (local + static_cast<std::size_t>(ncart) > sh.size()) {
            throw std::runtime_error("add_shell_values_cartesian: shell AO count mismatch");
        }

        double radial = 0.0;
        for (std::size_t p = 0; p < sh.alpha.size(); ++p) {
            radial += contr.coeff[p] * std::exp(-sh.alpha[p] * r2);
        }

        // Libint Cartesian ordering for angular momentum l:
        //   lx = l, l-1, ..., 0
        //   for each lx, ly = l-lx, ..., 0
        //   lz = l - lx - ly
        // Examples:
        //   p: x, y, z
        //   d: xx, xy, xz, yy, yz, zz
        //   f: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
        std::size_t k = 0;
        for (int lx = l; lx >= 0; --lx) {
            const int rem = l - lx;
            for (int ly = rem; ly >= 0; --ly) {
                const int lz = rem - ly;

                const double angular =
                    int_power(x, lx) * int_power(y, ly) * int_power(z, lz);

                phi(static_cast<Eigen::Index>(bf0 + local + k)) =
                    angular * radial;
                ++k;
            }
        }

        if (static_cast<int>(k) != ncart) {
            throw std::runtime_error("add_shell_values_cartesian: internal Cartesian count mismatch");
        }

        local += static_cast<std::size_t>(ncart);
    }

    if (local != sh.size()) {
        throw std::runtime_error("add_shell_values_cartesian: local AO count mismatch");
    }
}


void add_shell_values_grad_cartesian(const libint2::Shell& sh,
                                     const Eigen::Vector3d& r,
                                     AOValuesGrad& ao,
                                     std::size_t bf0) {
    const Eigen::Vector3d A = shell_origin(sh);
    const double x = r(0) - A(0);
    const double y = r(1) - A(1);
    const double z = r(2) - A(2);
    const double r2 = x*x + y*y + z*z;

    std::size_t local = 0;

    for (const auto& contr : sh.contr) {
        const int l = contr.l;
        const int ncart = ncartesian(l);

        if (contr.pure) {
            std::ostringstream oss;
            oss << "eval_ao_values_grad: pure/spherical shell detected with l=" << l
                << ". The current DFT grid AO gradient evaluator supports Cartesian "
                << "Gaussian shells only. Use basis.set_pure(false), or add a "
                << "Libint-compatible Cartesian-to-spherical transform.";
            throw std::runtime_error(oss.str());
        }

        if (local + static_cast<std::size_t>(ncart) > sh.size()) {
            throw std::runtime_error("add_shell_values_grad_cartesian: shell AO count mismatch");
        }

        double radial = 0.0;
        double dradial_x = 0.0;
        double dradial_y = 0.0;
        double dradial_z = 0.0;

        for (std::size_t p = 0; p < sh.alpha.size(); ++p) {
            const double alpha = sh.alpha[p];
            const double coeff = contr.coeff[p];
            const double e = std::exp(-alpha * r2);
            const double ce = coeff * e;

            radial += ce;

            const double factor = -2.0 * alpha * ce;
            dradial_x += factor * x;
            dradial_y += factor * y;
            dradial_z += factor * z;
        }

        std::size_t k = 0;
        for (int lx = l; lx >= 0; --lx) {
            const int rem = l - lx;
            for (int ly = rem; ly >= 0; --ly) {
                const int lz = rem - ly;

                const double x_lx = int_power(x, lx);
                const double y_ly = int_power(y, ly);
                const double z_lz = int_power(z, lz);
                const double angular = x_lx * y_ly * z_lz;

                const double dangular_dx = d_int_power(x, lx) * y_ly * z_lz;
                const double dangular_dy = x_lx * d_int_power(y, ly) * z_lz;
                const double dangular_dz = x_lx * y_ly * d_int_power(z, lz);

                const Eigen::Index idx = static_cast<Eigen::Index>(bf0 + local + k);

                ao.phi(idx) = angular * radial;
                ao.dphidx(idx) = dangular_dx * radial + angular * dradial_x;
                ao.dphidy(idx) = dangular_dy * radial + angular * dradial_y;
                ao.dphidz(idx) = dangular_dz * radial + angular * dradial_z;

                ++k;
            }
        }

        if (static_cast<int>(k) != ncart) {
            throw std::runtime_error(
                "add_shell_values_grad_cartesian: internal Cartesian count mismatch");
        }

        local += static_cast<std::size_t>(ncart);
    }

    if (local != sh.size()) {
        throw std::runtime_error("add_shell_values_grad_cartesian: local AO count mismatch");
    }
}

std::vector<GridPoint>
make_uniform_grid(const std::vector<libint2::Atom>& atoms,
                  const UniformGridOptions& options) {
    if (atoms.empty()) {
        throw std::runtime_error("make_uniform_grid: empty atom list");
    }
    if (options.spacing <= 0.0) {
        throw std::runtime_error("make_uniform_grid: spacing must be positive");
    }
    if (options.padding <= 0.0) {
        throw std::runtime_error("make_uniform_grid: padding must be positive");
    }

    Eigen::Vector3d lo(atoms[0].x, atoms[0].y, atoms[0].z);
    Eigen::Vector3d hi = lo;

    for (const auto& a : atoms) {
        lo(0) = std::min(lo(0), a.x);
        lo(1) = std::min(lo(1), a.y);
        lo(2) = std::min(lo(2), a.z);

        hi(0) = std::max(hi(0), a.x);
        hi(1) = std::max(hi(1), a.y);
        hi(2) = std::max(hi(2), a.z);
    }

    lo.array() -= options.padding;
    hi.array() += options.padding;

    const int nx = static_cast<int>(std::floor((hi(0) - lo(0)) / options.spacing)) + 1;
    const int ny = static_cast<int>(std::floor((hi(1) - lo(1)) / options.spacing)) + 1;
    const int nz = static_cast<int>(std::floor((hi(2) - lo(2)) / options.spacing)) + 1;

    const std::size_t npts =
        static_cast<std::size_t>(nx) *
        static_cast<std::size_t>(ny) *
        static_cast<std::size_t>(nz);

    if (npts > options.max_points) {
        std::ostringstream oss;
        oss << "make_uniform_grid: too many grid points (" << npts
            << "), increase spacing or max_points";
        throw std::runtime_error(oss.str());
    }

    const double w = options.spacing * options.spacing * options.spacing;

    std::vector<GridPoint> grid;
    grid.reserve(npts);

    for (int ix = 0; ix < nx; ++ix) {
        const double x = lo(0) + ix * options.spacing;
        for (int iy = 0; iy < ny; ++iy) {
            const double y = lo(1) + iy * options.spacing;
            for (int iz = 0; iz < nz; ++iz) {
                const double z = lo(2) + iz * options.spacing;
                grid.push_back(GridPoint{Eigen::Vector3d{x, y, z}, w});
            }
        }
    }

    return grid;
}

Eigen::VectorXd
eval_ao_values(const libint2::BasisSet& basis,
               const Eigen::Vector3d& r) {
    const int nbf = static_cast<int>(basis.nbf());
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(nbf);

    const auto& shell2bf = basis.shell2bf();

    for (std::size_t s = 0; s < basis.size(); ++s) {
        add_shell_values_cartesian(basis[s], r, phi, shell2bf[s]);
    }

    return phi;
}

Eigen::VectorXd
eval_ao_values_sp_only(const libint2::BasisSet& basis,
                       const Eigen::Vector3d& r) {
    return eval_ao_values(basis, r);
}

AOValuesGrad
eval_ao_values_grad(const libint2::BasisSet& basis,
                    const Eigen::Vector3d& r) {
    const int nbf = static_cast<int>(basis.nbf());

    AOValuesGrad ao;
    ao.phi = Eigen::VectorXd::Zero(nbf);
    ao.dphidx = Eigen::VectorXd::Zero(nbf);
    ao.dphidy = Eigen::VectorXd::Zero(nbf);
    ao.dphidz = Eigen::VectorXd::Zero(nbf);

    const auto& shell2bf = basis.shell2bf();

    for (std::size_t s = 0; s < basis.size(); ++s) {
        add_shell_values_grad_cartesian(basis[s], r, ao, shell2bf[s]);
    }

    return ao;
}

DensityGradPoint
eval_density_gradient(const Eigen::MatrixXd& D,
                      const AOValuesGrad& ao) {
    const int nbf = static_cast<int>(ao.phi.size());

    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error("eval_density_gradient: density matrix dimension mismatch");
    }

    DensityGradPoint out;

    const Eigen::VectorXd Dphi = D * ao.phi;
    out.rho = ao.phi.dot(Dphi);

    // D is symmetric for RHF/RKS density matrices:
    // rho = phi^T D phi
    // d rho / dx = 2 (dphi/dx)^T D phi
    out.grad_rho(0) = 2.0 * ao.dphidx.dot(Dphi);
    out.grad_rho(1) = 2.0 * ao.dphidy.dot(Dphi);
    out.grad_rho(2) = 2.0 * ao.dphidz.dot(Dphi);

    out.sigma = out.grad_rho.squaredNorm();
    return out;
}

DensityGradPoint
eval_density_gradient(const libint2::BasisSet& basis,
                      const Eigen::MatrixXd& D,
                      const Eigen::Vector3d& r) {
    const AOValuesGrad ao = eval_ao_values_grad(basis, r);
    return eval_density_gradient(D, ao);
}

double
check_ao_gradient_finite_difference(const libint2::BasisSet& basis,
                                    const Eigen::Vector3d& r,
                                    double h,
                                    bool verbose) {
    if (!(h > 0.0)) {
        throw std::runtime_error("check_ao_gradient_finite_difference: h must be positive");
    }

    const AOValuesGrad ao = eval_ao_values_grad(basis, r);
    double max_err = 0.0;

    for (int idir = 0; idir < 3; ++idir) {
        Eigen::Vector3d rp = r;
        Eigen::Vector3d rm = r;
        rp(idir) += h;
        rm(idir) -= h;

        const Eigen::VectorXd phip = eval_ao_values(basis, rp);
        const Eigen::VectorXd phim = eval_ao_values(basis, rm);
        const Eigen::VectorXd fd = (phip - phim) / (2.0 * h);

        Eigen::VectorXd analytic;
        if (idir == 0) {
            analytic = ao.dphidx;
        } else if (idir == 1) {
            analytic = ao.dphidy;
        } else {
            analytic = ao.dphidz;
        }

        const double err = (analytic - fd).cwiseAbs().maxCoeff();
        max_err = std::max(max_err, err);

        if (verbose) {
            std::cout << "AO gradient finite-difference check dir "
                      << idir
                      << "  max_abs_err = " << std::scientific
                      << std::setprecision(6) << err << "\n";
        }
    }

    if (verbose) {
        std::cout << "AO gradient finite-difference check overall max_abs_err = "
                  << std::scientific << std::setprecision(6)
                  << max_err << "\n";
    }

    return max_err;
}

double
check_density_gradient_finite_difference(const libint2::BasisSet& basis,
                                         const Eigen::MatrixXd& D,
                                         const Eigen::Vector3d& r,
                                         double h,
                                         bool verbose) {
    if (!(h > 0.0)) {
        throw std::runtime_error("check_density_gradient_finite_difference: h must be positive");
    }

    const DensityGradPoint analytic = eval_density_gradient(basis, D, r);
    Eigen::Vector3d fd = Eigen::Vector3d::Zero();

    for (int idir = 0; idir < 3; ++idir) {
        Eigen::Vector3d rp = r;
        Eigen::Vector3d rm = r;
        rp(idir) += h;
        rm(idir) -= h;

        const Eigen::VectorXd phip = eval_ao_values(basis, rp);
        const Eigen::VectorXd phim = eval_ao_values(basis, rm);

        const double rhop = phip.dot(D * phip);
        const double rhom = phim.dot(D * phim);
        fd(idir) = (rhop - rhom) / (2.0 * h);
    }

    const double max_err = (analytic.grad_rho - fd).cwiseAbs().maxCoeff();

    if (verbose) {
        std::cout << "Density gradient finite-difference check\n"
                  << "  rho       = " << std::scientific << std::setprecision(12)
                  << analytic.rho << "\n"
                  << "  grad ana  = "
                  << analytic.grad_rho.transpose() << "\n"
                  << "  grad fd   = "
                  << fd.transpose() << "\n"
                  << "  sigma     = "
                  << analytic.sigma << "\n"
                  << "  max error = "
                  << max_err << "\n";
    }

    return max_err;
}


LDAExchangeResult
build_lda_exchange_only_sp(const libint2::BasisSet& basis,
                           const Eigen::MatrixXd& D,
                           const std::vector<GridPoint>& grid,
                           double rho_cutoff) {
    const int nbf = static_cast<int>(basis.nbf());
    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error("build_lda_exchange_only_sp: density matrix dimension mismatch");
    }

    const double Cx = 0.75 * std::pow(3.0 / pi, 1.0 / 3.0);
    const double vx_pref = std::pow(3.0 / pi, 1.0 / 3.0);

    LDAExchangeResult result;
    result.Vx = Eigen::MatrixXd::Zero(nbf, nbf);
    result.Vc = Eigen::MatrixXd::Zero(nbf, nbf);
    result.Vxc = Eigen::MatrixXd::Zero(nbf, nbf);

    // This straightforward implementation is intentionally simple.
    // It is not optimized for large grids.
    for (const auto& gp : grid) {
        const Eigen::VectorXd phi = eval_ao_values(basis, gp.r);

        double rho = phi.dot(D * phi);
        if (rho < 0.0 && rho > -1.0e-12) {
            rho = 0.0; // tolerate tiny numerical noise
        }
        if (rho <= rho_cutoff) {
            continue;
        }

        result.npoints_used += 1;
        result.Ne_grid += gp.w * rho;
        result.rho_max = std::max(result.rho_max, rho);

        const double rho_13 = std::cbrt(rho);
        const double rho_43 = rho * rho_13;

        const double fx = -Cx * rho_43;
        const double vx = -vx_pref * rho_13;

        result.Ex += gp.w * fx;
        result.Vx.noalias() += gp.w * vx * (phi * phi.transpose());
    }

    result.Vx = 0.5 * (result.Vx + result.Vx.transpose());
    result.Vc.setZero(nbf, nbf);
    result.Vxc = result.Vx;
    result.Ec = 0.0;
    result.Exc = result.Ex;
    return result;
}

LDAExchangeResult
build_lda_libxc_sp(const libint2::BasisSet& basis,
                   const Eigen::MatrixXd& D,
                   const std::vector<GridPoint>& grid,
                   LDAFunctional functional,
                   double rho_cutoff) {
    const int nbf = static_cast<int>(basis.nbf());
    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error("build_lda_libxc_sp: density matrix dimension mismatch");
    }

    const bool use_x =
        (functional == LDAFunctional::Libxc_LDA_X ||
         functional == LDAFunctional::Libxc_LDA_X_PZ81C);
    const bool use_c =
        (functional == LDAFunctional::Libxc_LDA_X_PZ81C);

    if (!use_x) {
        throw std::runtime_error(
            "build_lda_libxc_sp: unsupported functional " +
            lda_functional_name(functional));
    }

    LDAExchangeResult result;
    result.Vx = Eigen::MatrixXd::Zero(nbf, nbf);
    result.Vc = Eigen::MatrixXd::Zero(nbf, nbf);
    result.Vxc = Eigen::MatrixXd::Zero(nbf, nbf);

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif

    std::vector<Eigen::MatrixXd> Vx_threads;
    std::vector<Eigen::MatrixXd> Vc_threads;
    std::vector<double> Ex_threads(nthreads, 0.0);
    std::vector<double> Ec_threads(nthreads, 0.0);
    std::vector<double> Ne_threads(nthreads, 0.0);
    std::vector<double> rho_max_threads(nthreads, 0.0);
    std::vector<std::size_t> npoints_threads(nthreads, 0);

    Vx_threads.reserve(nthreads);
    Vc_threads.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        Vx_threads.push_back(Eigen::MatrixXd::Zero(nbf, nbf));
        Vc_threads.push_back(Eigen::MatrixXd::Zero(nbf, nbf));
    }

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif

        LibxcFunctionalHandle xc_x_thread(XC_LDA_X, XC_UNPOLARIZED);
        LibxcFunctionalHandle xc_c_thread(XC_LDA_C_PZ, XC_UNPOLARIZED);

        Eigen::MatrixXd& Vx_local = Vx_threads[tid];
        Eigen::MatrixXd& Vc_local = Vc_threads[tid];

        double Ex_local = 0.0;
        double Ec_local = 0.0;
        double Ne_local = 0.0;
        double rho_max_local = 0.0;
        std::size_t npoints_used_local = 0;

#pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t ig = 0;
             ig < static_cast<std::ptrdiff_t>(grid.size());
             ++ig) {
            const GridPoint& gp = grid[static_cast<std::size_t>(ig)];

            const Eigen::VectorXd phi = eval_ao_values(basis, gp.r);
            const Eigen::VectorXd Dphi = D * phi;

            double rho = phi.dot(Dphi);
            if (rho < 0.0 && rho > -1.0e-12) {
                rho = 0.0;
            }
            if (rho <= rho_cutoff) {
                continue;
            }

            npoints_used_local += 1;
            Ne_local += gp.w * rho;
            rho_max_local = std::max(rho_max_local, rho);

            const double rho_arr[1] = {rho};
            double eps_x[1] = {0.0};
            double v_x[1] = {0.0};
            double eps_c[1] = {0.0};
            double v_c[1] = {0.0};

            xc_lda_exc_vxc(xc_x_thread.get(), 1, rho_arr, eps_x, v_x);
            if (use_c) {
                xc_lda_exc_vxc(xc_c_thread.get(), 1, rho_arr, eps_c, v_c);
            }

            Ex_local += gp.w * rho * eps_x[0];
            Ec_local += gp.w * rho * eps_c[0];

            const Eigen::MatrixXd phiphi = phi * phi.transpose();
            Vx_local.noalias() += gp.w * v_x[0] * phiphi;
            if (use_c) {
                Vc_local.noalias() += gp.w * v_c[0] * phiphi;
            }
        }

        Ex_threads[tid] = Ex_local;
        Ec_threads[tid] = Ec_local;
        Ne_threads[tid] = Ne_local;
        rho_max_threads[tid] = rho_max_local;
        npoints_threads[tid] = npoints_used_local;
    }

    for (int t = 0; t < nthreads; ++t) {
        result.Vx.noalias() += Vx_threads[t];
        result.Vc.noalias() += Vc_threads[t];
        result.Ex += Ex_threads[t];
        result.Ec += Ec_threads[t];
        result.Ne_grid += Ne_threads[t];
        result.rho_max = std::max(result.rho_max, rho_max_threads[t]);
        result.npoints_used += npoints_threads[t];
    }

    result.Vx = 0.5 * (result.Vx + result.Vx.transpose());
    result.Vc = 0.5 * (result.Vc + result.Vc.transpose());
    result.Vxc = result.Vx + result.Vc;
    result.Exc = result.Ex + result.Ec;

    return result;
}

LDAExchangeResult
build_gga_libxc_sp(const libint2::BasisSet& basis,
                   const Eigen::MatrixXd& D,
                   const std::vector<GridPoint>& grid,
                   LDAFunctional functional,
                   double rho_cutoff) {
    const int nbf = static_cast<int>(basis.nbf());
    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error("build_gga_libxc_sp: density matrix dimension mismatch");
    }

    if (functional != LDAFunctional::Libxc_GGA_PBE) {
        throw std::runtime_error(
            "build_gga_libxc_sp: unsupported functional " +
            lda_functional_name(functional));
    }

    LDAExchangeResult result;
    result.Vx = Eigen::MatrixXd::Zero(nbf, nbf);
    result.Vc = Eigen::MatrixXd::Zero(nbf, nbf);
    result.Vxc = Eigen::MatrixXd::Zero(nbf, nbf);

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif

    std::vector<Eigen::MatrixXd> Vx_threads;
    std::vector<Eigen::MatrixXd> Vc_threads;
    std::vector<double> Ex_threads(nthreads, 0.0);
    std::vector<double> Ec_threads(nthreads, 0.0);
    std::vector<double> Ne_threads(nthreads, 0.0);
    std::vector<double> rho_max_threads(nthreads, 0.0);
    std::vector<double> sigma_max_threads(nthreads, 0.0);
    std::vector<std::size_t> npoints_threads(nthreads, 0);

    Vx_threads.reserve(nthreads);
    Vc_threads.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        Vx_threads.push_back(Eigen::MatrixXd::Zero(nbf, nbf));
        Vc_threads.push_back(Eigen::MatrixXd::Zero(nbf, nbf));
    }

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif

        LibxcFunctionalHandle xc_x_thread(XC_GGA_X_PBE, XC_UNPOLARIZED);
        LibxcFunctionalHandle xc_c_thread(XC_GGA_C_PBE, XC_UNPOLARIZED);

        Eigen::MatrixXd& Vx_local = Vx_threads[tid];
        Eigen::MatrixXd& Vc_local = Vc_threads[tid];

        double Ex_local = 0.0;
        double Ec_local = 0.0;
        double Ne_local = 0.0;
        double rho_max_local = 0.0;
        double sigma_max_local = 0.0;
        std::size_t npoints_used_local = 0;

#pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t ig = 0;
             ig < static_cast<std::ptrdiff_t>(grid.size());
             ++ig) {
            const GridPoint& gp = grid[static_cast<std::size_t>(ig)];

            const AOValuesGrad ao = eval_ao_values_grad(basis, gp.r);
            const DensityGradPoint den = eval_density_gradient(D, ao);

            double rho = den.rho;
            if (rho < 0.0 && rho > -1.0e-12) {
                rho = 0.0;
            }
            if (rho <= rho_cutoff) {
                continue;
            }

            const double sigma = std::max(0.0, den.sigma);

            npoints_used_local += 1;
            Ne_local += gp.w * rho;
            rho_max_local = std::max(rho_max_local, rho);
            sigma_max_local = std::max(sigma_max_local, sigma);

            const double rho_arr[1] = {rho};
            const double sigma_arr[1] = {sigma};

            double eps_x[1] = {0.0};
            double vrho_x[1] = {0.0};
            double vsigma_x[1] = {0.0};

            double eps_c[1] = {0.0};
            double vrho_c[1] = {0.0};
            double vsigma_c[1] = {0.0};

            xc_gga_exc_vxc(xc_x_thread.get(), 1, rho_arr, sigma_arr,
                           eps_x, vrho_x, vsigma_x);
            xc_gga_exc_vxc(xc_c_thread.get(), 1, rho_arr, sigma_arr,
                           eps_c, vrho_c, vsigma_c);

            Ex_local += gp.w * rho * eps_x[0];
            Ec_local += gp.w * rho * eps_c[0];

            const Eigen::VectorXd& phi = ao.phi;
            const Eigen::MatrixXd phiphi = phi * phi.transpose();

            // gdot(mu) = grad phi_mu dot grad rho.
            const Eigen::VectorXd gdot =
                den.grad_rho(0) * ao.dphidx +
                den.grad_rho(1) * ao.dphidy +
                den.grad_rho(2) * ao.dphidz;

            const Eigen::MatrixXd grad_term =
                gdot * phi.transpose() + phi * gdot.transpose();

            // For unpolarized GGA with sigma = grad rho · grad rho:
            // V_mn = int [ vrho phi_m phi_n
            //            + 2 vsigma grad rho · grad(phi_m phi_n) ] dr.
            Vx_local.noalias() += gp.w * vrho_x[0] * phiphi;
            Vx_local.noalias() += gp.w * 2.0 * vsigma_x[0] * grad_term;

            Vc_local.noalias() += gp.w * vrho_c[0] * phiphi;
            Vc_local.noalias() += gp.w * 2.0 * vsigma_c[0] * grad_term;
        }

        Ex_threads[tid] = Ex_local;
        Ec_threads[tid] = Ec_local;
        Ne_threads[tid] = Ne_local;
        rho_max_threads[tid] = rho_max_local;
        sigma_max_threads[tid] = sigma_max_local;
        npoints_threads[tid] = npoints_used_local;
    }

    for (int t = 0; t < nthreads; ++t) {
        result.Vx.noalias() += Vx_threads[t];
        result.Vc.noalias() += Vc_threads[t];
        result.Ex += Ex_threads[t];
        result.Ec += Ec_threads[t];
        result.Ne_grid += Ne_threads[t];
        result.rho_max = std::max(result.rho_max, rho_max_threads[t]);
        result.sigma_max = std::max(result.sigma_max, sigma_max_threads[t]);
        result.npoints_used += npoints_threads[t];
    }

    result.Vx = 0.5 * (result.Vx + result.Vx.transpose());
    result.Vc = 0.5 * (result.Vc + result.Vc.transpose());
    result.Vxc = result.Vx + result.Vc;
    result.Exc = result.Ex + result.Ec;

    return result;
}

LDAExchangeResult
build_lda_xc_sp(const libint2::BasisSet& basis,
                const Eigen::MatrixXd& D,
                const std::vector<GridPoint>& grid,
                LDAFunctional functional,
                double rho_cutoff) {
    switch (functional) {
        case LDAFunctional::SlaterX:
            return build_lda_exchange_only_sp(basis, D, grid, rho_cutoff);

        case LDAFunctional::Libxc_LDA_X:
        case LDAFunctional::Libxc_LDA_X_PZ81C:
            return build_lda_libxc_sp(basis, D, grid, functional, rho_cutoff);

        case LDAFunctional::Libxc_GGA_PBE:
            return build_gga_libxc_sp(basis, D, grid, functional, rho_cutoff);

        default:
            throw std::runtime_error("build_lda_xc_sp: unknown functional");
    }
}

Eigen::MatrixXd
build_j_direct(const libint2::BasisSet& basis,
               const Eigen::MatrixXd& D) {
    const int nbf = static_cast<int>(basis.nbf());
    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error("build_j_direct: density matrix dimension mismatch");
    }

    const int nshell = static_cast<int>(basis.size());
    const auto& shell2bf = basis.shell2bf();

    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nbf, nbf);

    const long long npairs =
        static_cast<long long>(nshell) * static_cast<long long>(nshell);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        libint2::Engine engine(libint2::Operator::coulomb,
                               basis.max_nprim(),
                               basis.max_l());

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (long long pair12 = 0; pair12 < npairs; ++pair12) {
            const int s1 = static_cast<int>(pair12 / nshell);
            const int s2 = static_cast<int>(pair12 % nshell);

            const std::size_t bf1 = shell2bf[s1];
            const std::size_t bf2 = shell2bf[s2];

            const std::size_t n1 = basis[s1].size();
            const std::size_t n2 = basis[s2].size();

            for (int s3 = 0; s3 < nshell; ++s3) {
                for (int s4 = 0; s4 < nshell; ++s4) {
                    engine.compute2<libint2::Operator::coulomb, libint2::BraKet::xx_xx, 0>(basis[s1], basis[s2], basis[s3], basis[s4]);

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

                                    const std::size_t idx =
                                        ((i * n2 + j) * n3 + k) * n4 + l;

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

double trace_product(const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& B) {
    return (A.array() * B.array()).sum();
}

RKSLDAResult
run_rks_lda_exchange_only_sp(const libint2::BasisSet& basis,
                             const Eigen::MatrixXd& S,
                             const Eigen::MatrixXd& Hcore,
                             int nelec,
                             double Enuc,
                             const std::vector<GridPoint>& grid,
                             const RKSLDAOptions& options) {
    const int nbf = static_cast<int>(basis.nbf());

    if (nelec <= 0 || nelec % 2 != 0) {
        throw std::runtime_error("run_rks_lda_exchange_only_sp: RKS requires positive even nelec");
    }
    if (S.rows() != nbf || S.cols() != nbf ||
        Hcore.rows() != nbf || Hcore.cols() != nbf) {
        throw std::runtime_error("run_rks_lda_exchange_only_sp: matrix dimension mismatch");
    }
    if (options.diis_start < 1) {
        throw std::runtime_error("run_rks_lda_exchange_only_sp: diis_start must be >= 1");
    }

    const int nocc = nelec / 2;
    const Eigen::MatrixXd X = symmetric_orthogonalization(S);

    // Core-Hamiltonian initial guess.
    Eigen::VectorXd eps;
    Eigen::MatrixXd C;
    diagonalize_fock(Hcore, X, eps, C);
    Eigen::MatrixXd D = build_density(C, nocc);

    DIISManager diis(std::max<std::size_t>(2, options.diis_max_vecs));

    double E_prev = std::numeric_limits<double>::quiet_NaN();

    RKSLDAResult out;
    out.E_nuc = Enuc;

    if (options.verbose) {
        std::cout << "RKS-LDA functional = "
                  << lda_functional_name(options.functional) << "\n";
    }

    for (int iter = 1; iter <= options.max_iter; ++iter) {
        const Eigen::MatrixXd J = build_j_direct(basis, D);
        const LDAExchangeResult xc = build_lda_xc_sp(basis, D, grid, options.functional);

        const Eigen::MatrixXd F_raw = Hcore + J + xc.Vxc;

        // The energy is evaluated from the density and the unextrapolated
        // KS ingredients generated by that density. DIIS is only used to
        // produce a better Fock matrix for diagonalization.
        const double E_elec =
            trace_product(D, Hcore) +
            0.5 * trace_product(D, J) +
            xc.Exc;

        const double E_total = E_elec + Enuc;

        Eigen::MatrixXd F_diag = F_raw;
        double diis_err_norm = std::numeric_limits<double>::quiet_NaN();
        bool used_diis = false;

        if (options.use_diis) {
            const Eigen::MatrixXd err =
                DIISManager::error_matrix(F_raw, D, S, X);
            diis_err_norm = err.norm();
            diis.add(F_raw, err);

            if (iter >= options.diis_start && diis.size() >= 2) {
                try {
                    F_diag = diis.extrapolate();
                    used_diis = true;
                } catch (const std::exception& e) {
                    // DIIS can occasionally become singular if stored error
                    // vectors are nearly linearly dependent. Falling back to
                    // the raw Fock is safer than aborting the SCF.
                    if (options.verbose) {
                        std::cout << "  DIIS skipped: " << e.what() << "\n";
                    }
                    F_diag = F_raw;
                    used_diis = false;
                }
            }
        }

        Eigen::VectorXd eps_new;
        Eigen::MatrixXd C_new;
        diagonalize_fock(F_diag, X, eps_new, C_new);

        Eigen::MatrixXd D_new = build_density(C_new, nocc);

        if (options.density_mixing > 0.0) {
            const double a = options.density_mixing;
            D_new = (1.0 - a) * D_new + a * D;
            D_new = 0.5 * (D_new + D_new.transpose());
        }

        const double dE =
            std::isfinite(E_prev) ? (E_total - E_prev) : 0.0;
        const double rmsD =
            (D_new - D).norm() / static_cast<double>(nbf);

        if (options.verbose) {
            std::cout << "RKS-LDA iter " << std::setw(3) << iter
                      << "  E = " << std::fixed << std::setprecision(12) << E_total
                      << "  dE = " << std::scientific << std::setprecision(3) << dE
                      << "  rmsD = " << rmsD;

            if (options.use_diis) {
                std::cout << "  diis_err = " << diis_err_norm
                          << "  DIIS = " << (used_diis ? "yes" : "no");
            }

            std::cout << "  Ex = " << xc.Ex
                      << "  Ec = " << xc.Ec
                      << "  Exc = " << xc.Exc
                      << "  Ne(grid) = " << xc.Ne_grid
                      << std::defaultfloat << "\n";
        }

        out.niter = iter;
        out.E_total = E_total;
        out.E_elec = E_elec;
        out.E_x = xc.Ex;
        out.E_c = xc.Ec;
        out.E_xc = xc.Exc;
        out.Ne_grid = xc.Ne_grid;
        out.F = F_raw;

        const bool e_ok = std::isfinite(E_prev) && std::abs(dE) < options.e_conv;
        const bool d_ok = rmsD < options.d_conv;

        D = D_new;
        C = C_new;
        eps = eps_new;
        E_prev = E_total;

        if (e_ok && d_ok) {
            out.converged = true;
            break;
        }
    }

    // Rebuild final F and energy from the final iterated density. Use the
    // same density in H/J/Ex to keep the reported energy internally consistent.
    const Eigen::MatrixXd J_final = build_j_direct(basis, D);
    const LDAExchangeResult xc_final = build_lda_xc_sp(basis, D, grid, options.functional);
    const Eigen::MatrixXd F_final = Hcore + J_final + xc_final.Vxc;

    Eigen::VectorXd eps_final;
    Eigen::MatrixXd C_final;
    diagonalize_fock(F_final, X, eps_final, C_final);

    const double E_elec_final =
        trace_product(D, Hcore) +
        0.5 * trace_product(D, J_final) +
        xc_final.Exc;

    out.E_elec = E_elec_final;
    out.E_total = E_elec_final + Enuc;
    out.E_x = xc_final.Ex;
    out.E_c = xc_final.Ec;
    out.E_xc = xc_final.Exc;
    out.Ne_grid = xc_final.Ne_grid;
    out.eps = eps_final;
    out.C = C_final;
    out.D = D;
    out.F = F_final;

    return out;
}

} // namespace dft
