#include "dft_lda.hpp"

#include "two_body_fock.hpp"
#include "xc_evaluator.hpp"
#include "xc_functional.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace dft {

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

std::string lda_functional_name(LDAFunctional functional) {
    switch (functional) {
        case LDAFunctional::SlaterX:
            return "hand-written Slater X";
        case LDAFunctional::Libxc_LDA_X:
            return "Libxc LDA_X";
        case LDAFunctional::Libxc_LDA_X_PZ81C:
            return "Libxc LDA_X + LDA_C_PZ";
        case LDAFunctional::Libxc_GGA_PBE:
            return "Libxc PBE GGA";
    }
    return "unknown XC functional";
}

miniqc::xc::XCFunctional legacy_to_xc_functional(LDAFunctional functional) {
    switch (functional) {
        case LDAFunctional::SlaterX:
        case LDAFunctional::Libxc_LDA_X:
            return miniqc::xc::make_xc_functional("lda_x");
        case LDAFunctional::Libxc_LDA_X_PZ81C:
            return miniqc::xc::make_xc_functional("lda_x_pz81");
        case LDAFunctional::Libxc_GGA_PBE:
            return miniqc::xc::make_xc_functional("pbe");
    }
    throw std::runtime_error("unsupported legacy LDAFunctional");
}

Eigen::Vector3d shell_origin(const libint2::Shell& sh) {
    return Eigen::Vector3d{sh.O[0], sh.O[1], sh.O[2]};
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

void check_density_shape(const libint2::BasisSet& basis,
                         const Eigen::MatrixXd& D,
                         const char* caller) {
    const int nbf = static_cast<int>(basis.nbf());
    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error(std::string(caller) + ": density matrix dimension mismatch");
    }
}

void add_shell_values_cartesian(const libint2::Shell& sh,
                                const Eigen::Vector3d& r,
                                Eigen::VectorXd& phi,
                                std::size_t bf0) {
    const Eigen::Vector3d A = shell_origin(sh);
    const double x = r(0) - A(0);
    const double y = r(1) - A(1);
    const double z = r(2) - A(2);
    const double r2 = x * x + y * y + z * z;

    std::size_t local = 0;
    for (const auto& contr : sh.contr) {
        const int l = contr.l;
        const int ncart = ncartesian(l);

        if (contr.pure) {
            std::ostringstream oss;
            oss << "eval_ao_values: pure/spherical shell detected with l=" << l
                << ". The current DFT grid AO evaluator supports Cartesian Gaussian shells only.";
            throw std::runtime_error(oss.str());
        }
        if (local + static_cast<std::size_t>(ncart) > sh.size()) {
            throw std::runtime_error("add_shell_values_cartesian: shell AO count mismatch");
        }

        double radial = 0.0;
        for (std::size_t p = 0; p < sh.alpha.size(); ++p) {
            radial += contr.coeff[p] * std::exp(-sh.alpha[p] * r2);
        }

        std::size_t k = 0;
        for (int lx = l; lx >= 0; --lx) {
            const int rem = l - lx;
            for (int ly = rem; ly >= 0; --ly) {
                const int lz = rem - ly;
                const double angular =
                    int_power(x, lx) * int_power(y, ly) * int_power(z, lz);
                phi(static_cast<Eigen::Index>(bf0 + local + k)) = angular * radial;
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
    const double r2 = x * x + y * y + z * z;

    std::size_t local = 0;
    for (const auto& contr : sh.contr) {
        const int l = contr.l;
        const int ncart = ncartesian(l);

        if (contr.pure) {
            std::ostringstream oss;
            oss << "eval_ao_values_grad: pure/spherical shell detected with l=" << l
                << ". The current DFT grid AO gradient evaluator supports Cartesian Gaussian shells only.";
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
            throw std::runtime_error("add_shell_values_grad_cartesian: internal Cartesian count mismatch");
        }
        local += static_cast<std::size_t>(ncart);
    }

    if (local != sh.size()) {
        throw std::runtime_error("add_shell_values_grad_cartesian: local AO count mismatch");
    }
}

void accumulate_legacy_evaluator_point(const Eigen::VectorXd& phi,
                                       double weight,
                                       double rho,
                                       const miniqc::xc::XCOutputPoint& xc,
                                       LDAExchangeResult& result) {
    result.Exc += weight * rho * xc.exc;
    result.Vxc.noalias() += weight * xc.vrho * (phi * phi.transpose());
}

void accumulate_legacy_evaluator_point(const AOValuesGrad& ao,
                                       const DensityGradPoint& den,
                                       double weight,
                                       double rho,
                                       const miniqc::xc::XCOutputPoint& xc,
                                       LDAExchangeResult& result) {
    const Eigen::VectorXd& phi = ao.phi;
    const Eigen::MatrixXd phiphi = phi * phi.transpose();
    const Eigen::VectorXd gdot =
        den.grad_rho(0) * ao.dphidx +
        den.grad_rho(1) * ao.dphidy +
        den.grad_rho(2) * ao.dphidz;
    const Eigen::MatrixXd grad_term =
        gdot * phi.transpose() + phi * gdot.transpose();

    result.Exc += weight * rho * xc.exc;
    result.Vxc.noalias() += weight * xc.vrho * phiphi;
    result.Vxc.noalias() += weight * 2.0 * xc.vsigma * grad_term;
}

LDAExchangeResult build_semilocal_with_evaluator(const libint2::BasisSet& basis,
                                                 const Eigen::MatrixXd& D,
                                                 const std::vector<GridPoint>& grid,
                                                 const miniqc::xc::XCFunctional& functional,
                                                 double rho_cutoff) {
    check_density_shape(basis, D, "build_semilocal_with_evaluator");
    if (functional.is_meta_gga()) {
        throw std::runtime_error("build_semilocal_with_evaluator: meta-GGA is not implemented here");
    }

    const int nbf = static_cast<int>(basis.nbf());
    LDAExchangeResult result;
    result.Vx = Eigen::MatrixXd::Zero(nbf, nbf);
    result.Vc = Eigen::MatrixXd::Zero(nbf, nbf);
    result.Vxc = Eigen::MatrixXd::Zero(nbf, nbf);

    for (const GridPoint& gp : grid) {
        if (functional.requirements.needs_sigma) {
            const AOValuesGrad ao = eval_ao_values_grad(basis, gp.r);
            DensityGradPoint den = eval_density_gradient(D, ao);
            double rho = den.rho;
            if (rho < 0.0 && rho > -1.0e-12) rho = 0.0;
            if (rho <= rho_cutoff) continue;
            den.sigma = std::max(0.0, den.sigma);

            miniqc::xc::XCInputPoint input;
            input.rho = rho;
            input.sigma = den.sigma;
            const miniqc::xc::XCOutputPoint xc = miniqc::xc::evaluate_xc_point(functional, input);
            accumulate_legacy_evaluator_point(ao, den, gp.w, rho, xc, result);

            result.Ne_grid += gp.w * rho;
            result.rho_max = std::max(result.rho_max, rho);
            result.sigma_max = std::max(result.sigma_max, den.sigma);
            result.npoints_used += 1;
        } else {
            const Eigen::VectorXd phi = eval_ao_values(basis, gp.r);
            double rho = phi.dot(D * phi);
            if (rho < 0.0 && rho > -1.0e-12) rho = 0.0;
            if (rho <= rho_cutoff) continue;

            miniqc::xc::XCInputPoint input;
            input.rho = rho;
            const miniqc::xc::XCOutputPoint xc = miniqc::xc::evaluate_xc_point(functional, input);
            accumulate_legacy_evaluator_point(phi, gp.w, rho, xc, result);

            result.Ne_grid += gp.w * rho;
            result.rho_max = std::max(result.rho_max, rho);
            result.npoints_used += 1;
        }
    }

    result.Vxc = 0.5 * (result.Vxc + result.Vxc.transpose());
    result.Vx = result.Vxc;
    result.Vc.setZero(nbf, nbf);
    result.Ex = result.Exc;
    result.Ec = 0.0;
    return result;
}

}  // namespace

std::vector<GridPoint>
make_uniform_grid(const std::vector<libint2::Atom>& atoms,
                  const UniformGridOptions& options) {
    if (atoms.empty()) throw std::runtime_error("make_uniform_grid: empty atom list");
    if (options.spacing <= 0.0) throw std::runtime_error("make_uniform_grid: spacing must be positive");
    if (options.padding <= 0.0) throw std::runtime_error("make_uniform_grid: padding must be positive");

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
    const std::size_t npts = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    if (npts > options.max_points) {
        std::ostringstream oss;
        oss << "make_uniform_grid: too many grid points (" << npts << ")";
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

Eigen::VectorXd eval_ao_values(const libint2::BasisSet& basis,
                               const Eigen::Vector3d& r) {
    const int nbf = static_cast<int>(basis.nbf());
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(nbf);
    const auto& shell2bf = basis.shell2bf();
    for (std::size_t s = 0; s < basis.size(); ++s) {
        add_shell_values_cartesian(basis[s], r, phi, shell2bf[s]);
    }
    return phi;
}

Eigen::VectorXd eval_ao_values_sp_only(const libint2::BasisSet& basis,
                                       const Eigen::Vector3d& r) {
    return eval_ao_values(basis, r);
}

AOValuesGrad eval_ao_values_grad(const libint2::BasisSet& basis,
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

DensityGradPoint eval_density_gradient(const Eigen::MatrixXd& D,
                                       const AOValuesGrad& ao) {
    const int nbf = static_cast<int>(ao.phi.size());
    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error("eval_density_gradient: density matrix dimension mismatch");
    }

    DensityGradPoint out;
    const Eigen::VectorXd Dphi = D * ao.phi;
    out.rho = ao.phi.dot(Dphi);
    out.grad_rho(0) = 2.0 * ao.dphidx.dot(Dphi);
    out.grad_rho(1) = 2.0 * ao.dphidy.dot(Dphi);
    out.grad_rho(2) = 2.0 * ao.dphidz.dot(Dphi);
    out.sigma = out.grad_rho.squaredNorm();
    return out;
}

DensityGradPoint eval_density_gradient(const libint2::BasisSet& basis,
                                       const Eigen::MatrixXd& D,
                                       const Eigen::Vector3d& r) {
    const AOValuesGrad ao = eval_ao_values_grad(basis, r);
    return eval_density_gradient(D, ao);
}

double check_ao_gradient_finite_difference(const libint2::BasisSet& basis,
                                           const Eigen::Vector3d& r,
                                           double h,
                                           bool verbose) {
    if (!(h > 0.0)) throw std::runtime_error("check_ao_gradient_finite_difference: h must be positive");
    const AOValuesGrad ao = eval_ao_values_grad(basis, r);
    double max_err = 0.0;
    for (int idir = 0; idir < 3; ++idir) {
        Eigen::Vector3d rp = r;
        Eigen::Vector3d rm = r;
        rp(idir) += h;
        rm(idir) -= h;
        const Eigen::VectorXd fd = (eval_ao_values(basis, rp) - eval_ao_values(basis, rm)) / (2.0 * h);
        const Eigen::VectorXd analytic = (idir == 0) ? ao.dphidx : (idir == 1 ? ao.dphidy : ao.dphidz);
        const double err = (analytic - fd).cwiseAbs().maxCoeff();
        max_err = std::max(max_err, err);
        if (verbose) {
            std::cout << "AO gradient finite-difference check dir " << idir
                      << "  max_abs_err = " << std::scientific << std::setprecision(6)
                      << err << "\n";
        }
    }
    return max_err;
}

double check_density_gradient_finite_difference(const libint2::BasisSet& basis,
                                                const Eigen::MatrixXd& D,
                                                const Eigen::Vector3d& r,
                                                double h,
                                                bool verbose) {
    if (!(h > 0.0)) throw std::runtime_error("check_density_gradient_finite_difference: h must be positive");
    const DensityGradPoint analytic = eval_density_gradient(basis, D, r);
    Eigen::Vector3d fd = Eigen::Vector3d::Zero();
    for (int idir = 0; idir < 3; ++idir) {
        Eigen::Vector3d rp = r;
        Eigen::Vector3d rm = r;
        rp(idir) += h;
        rm(idir) -= h;
        const Eigen::VectorXd phip = eval_ao_values(basis, rp);
        const Eigen::VectorXd phim = eval_ao_values(basis, rm);
        fd(idir) = (phip.dot(D * phip) - phim.dot(D * phim)) / (2.0 * h);
    }
    const double max_err = (analytic.grad_rho - fd).cwiseAbs().maxCoeff();
    if (verbose) {
        std::cout << "Density gradient finite-difference check max error = "
                  << std::scientific << std::setprecision(6) << max_err << "\n";
    }
    return max_err;
}

LDAExchangeResult build_lda_exchange_only_sp(const libint2::BasisSet& basis,
                                             const Eigen::MatrixXd& D,
                                             const std::vector<GridPoint>& grid,
                                             double rho_cutoff) {
    check_density_shape(basis, D, "build_lda_exchange_only_sp");
    const int nbf = static_cast<int>(basis.nbf());
    const double Cx = 0.75 * std::pow(3.0 / pi, 1.0 / 3.0);
    const double vx_pref = std::pow(3.0 / pi, 1.0 / 3.0);

    LDAExchangeResult result;
    result.Vx = Eigen::MatrixXd::Zero(nbf, nbf);
    result.Vc = Eigen::MatrixXd::Zero(nbf, nbf);
    result.Vxc = Eigen::MatrixXd::Zero(nbf, nbf);

    for (const GridPoint& gp : grid) {
        const Eigen::VectorXd phi = eval_ao_values(basis, gp.r);
        double rho = phi.dot(D * phi);
        if (rho < 0.0 && rho > -1.0e-12) rho = 0.0;
        if (rho <= rho_cutoff) continue;
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

LDAExchangeResult build_lda_libxc_sp(const libint2::BasisSet& basis,
                                     const Eigen::MatrixXd& D,
                                     const std::vector<GridPoint>& grid,
                                     LDAFunctional functional,
                                     double rho_cutoff) {
    if (functional != LDAFunctional::Libxc_LDA_X &&
        functional != LDAFunctional::Libxc_LDA_X_PZ81C) {
        throw std::runtime_error("build_lda_libxc_sp: unsupported functional " + lda_functional_name(functional));
    }
    return build_semilocal_with_evaluator(basis, D, grid, legacy_to_xc_functional(functional), rho_cutoff);
}

LDAExchangeResult build_gga_libxc_sp(const libint2::BasisSet& basis,
                                     const Eigen::MatrixXd& D,
                                     const std::vector<GridPoint>& grid,
                                     LDAFunctional functional,
                                     double rho_cutoff) {
    if (functional != LDAFunctional::Libxc_GGA_PBE) {
        throw std::runtime_error("build_gga_libxc_sp: unsupported functional " + lda_functional_name(functional));
    }
    return build_semilocal_with_evaluator(basis, D, grid, legacy_to_xc_functional(functional), rho_cutoff);
}

LDAExchangeResult build_lda_xc_sp(const libint2::BasisSet& basis,
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
    }
    throw std::runtime_error("build_lda_xc_sp: unknown functional");
}

Eigen::MatrixXd build_j_direct(const libint2::BasisSet& basis,
                               const Eigen::MatrixXd& D) {
    return miniqc::build_j_direct(basis, D);
}

double trace_product(const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::runtime_error("trace_product: dimension mismatch");
    }
    return (A.array() * B.array()).sum();
}

} // namespace dft
