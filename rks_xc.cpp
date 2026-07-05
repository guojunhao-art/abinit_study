#include "rks_xc.hpp"

#include "dft_xc_matrix.hpp"
#include "linalg_utils.hpp"
#include "scf_diis.hpp"
#include "two_body_fock.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace miniqc::rks {

namespace {

void validate_inputs(const libint2::BasisSet& basis,
                     const Eigen::MatrixXd& S,
                     const Eigen::MatrixXd& Hcore,
                     int nelec,
                     const xc::XCFunctional& functional,
                     const RKSXCOptions& options) {
    const int nbf = static_cast<int>(basis.nbf());
    if (S.rows() != nbf || S.cols() != nbf ||
        Hcore.rows() != nbf || Hcore.cols() != nbf) {
        throw std::runtime_error("run_rks_xc: matrix dimension mismatch");
    }
    if (nelec <= 0 || nelec % 2 != 0) {
        throw std::runtime_error("run_rks_xc: only positive even-electron closed shells are supported");
    }
    if (nelec / 2 > nbf) {
        throw std::runtime_error("run_rks_xc: more occupied orbitals than basis functions");
    }
    if (functional.spin_mode != xc::SpinMode::Restricted) {
        throw std::runtime_error("run_rks_xc: only restricted functionals are supported");
    }
    if (functional.is_range_separated) {
        throw std::runtime_error("run_rks_xc: range-separated hybrids are not implemented yet");
    }
    if (functional.is_meta_gga()) {
        throw std::runtime_error("run_rks_xc: meta-GGA tau/vtau support is not implemented yet");
    }
    if (options.max_iter <= 0) {
        throw std::runtime_error("run_rks_xc: max_iter must be positive");
    }
    if (options.density_mixing < 0.0 || options.density_mixing >= 1.0) {
        throw std::runtime_error("run_rks_xc: density_mixing must satisfy 0 <= mixing < 1");
    }
}

Eigen::MatrixXd maybe_apply_density_mixing(const Eigen::MatrixXd& D_new,
                                           const Eigen::MatrixXd& D_old,
                                           double mixing) {
    if (mixing == 0.0) return D_new;
    return (1.0 - mixing) * D_new + mixing * D_old;
}

}  // namespace

RKSXCResult run_rks_xc(const libint2::BasisSet& basis,
                       const Eigen::MatrixXd& S,
                       const Eigen::MatrixXd& Hcore,
                       int nelec,
                       double Enuc,
                       const std::vector<dft::GridPoint>& grid,
                       const xc::XCFunctional& functional,
                       const RKSXCOptions& options) {
    validate_inputs(basis, S, Hcore, nelec, functional, options);

    const int nocc = nelec / 2;
    const int nbf = static_cast<int>(basis.nbf());

    const Eigen::MatrixXd X = symmetric_orthogonalization(S);

    Eigen::VectorXd eps;
    Eigen::MatrixXd C;
    diagonalize_fock(Hcore, X, eps, C);
    Eigen::MatrixXd D = build_density(C, nocc);

    scf::DIISManager diis(options.diis_max_vecs);

    RKSXCResult result;
    result.E_nuc = Enuc;
    result.eps = eps;
    result.C = C;
    result.D = D;
    result.F = Hcore;

    double E_old = std::numeric_limits<double>::infinity();

    if (options.verbose) {
        std::cout << "\n=== Total-XC RKS SCF ===\n"
                  << "functional = " << functional.name << "\n"
                  << "hybrid ax  = " << functional.exact_exchange_fraction << "\n"
                  << "iter          E_total              dE             dD\n";
    }

    for (int iter = 1; iter <= options.max_iter; ++iter) {
        const Eigen::MatrixXd J = build_j_direct(basis, D);
        const dft::LDAExchangeResult xc_result =
            dft::build_xc_matrix_with_evaluator_sp(
                basis, D, grid, functional, options.rho_cutoff);

        Eigen::MatrixXd K = Eigen::MatrixXd::Zero(nbf, nbf);
        double E_exact_exchange = 0.0;

        Eigen::MatrixXd F = Hcore + J + xc_result.Vxc;
        if (functional.is_hybrid()) {
            K = build_k_direct(basis, D);
            F.noalias() -= 0.5 * functional.exact_exchange_fraction * K;
            E_exact_exchange = hybrid_exact_exchange_energy(
                D, K, functional.exact_exchange_fraction);
        }
        F = 0.5 * (F + F.transpose());

        const double E_one = trace_product(D, Hcore);
        const double E_coulomb = 0.5 * trace_product(D, J);
        const double E_elec = E_one + E_coulomb + xc_result.Exc + E_exact_exchange;
        const double E_total = E_elec + Enuc;
        const double dE = (std::isfinite(E_old)) ? E_total - E_old : E_total;

        Eigen::MatrixXd F_diagonalized = F;
        if (options.use_diis && iter >= options.diis_start) {
            const Eigen::MatrixXd err = scf::DIISManager::error_matrix(F, D, S, X);
            diis.add(F, err);
            if (diis.size() >= 2) {
                F_diagonalized = diis.extrapolate();
            }
        }

        Eigen::VectorXd eps_new;
        Eigen::MatrixXd C_new;
        diagonalize_fock(F_diagonalized, X, eps_new, C_new);
        Eigen::MatrixXd D_new = build_density(C_new, nocc);
        D_new = maybe_apply_density_mixing(D_new, D, options.density_mixing);
        D_new = 0.5 * (D_new + D_new.transpose());

        const double dD = (D_new - D).cwiseAbs().maxCoeff();

        if (options.verbose) {
            std::cout << std::setw(4) << iter << "  "
                      << std::fixed << std::setprecision(12) << E_total << "  "
                      << std::scientific << std::setprecision(3) << dE << "  "
                      << dD << "\n";
        }

        result.niter = iter;
        result.E_total = E_total;
        result.E_elec = E_elec;
        result.E_one = E_one;
        result.E_coulomb = E_coulomb;
        result.E_xc = xc_result.Exc;
        result.E_exact_exchange = E_exact_exchange;
        result.E_nuc = Enuc;
        result.Ne_grid = xc_result.Ne_grid;
        result.eps = eps_new;
        result.C = C_new;
        result.D = D_new;
        result.F = F;

        const bool converged =
            (iter > 1) &&
            (std::abs(dE) < options.e_conv) &&
            (dD < options.d_conv);
        if (converged) {
            result.converged = true;
            break;
        }

        E_old = E_total;
        D = D_new;
    }

    return result;
}

}  // namespace miniqc::rks
