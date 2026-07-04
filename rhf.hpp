#pragma once

#include "core_molecule.hpp"
#include "linalg_utils.hpp"
#include "one_body_integrals.hpp"
#include "scf_diis.hpp"
#include "two_body_fock.hpp"

#include <Eigen/Dense>
#include <libint2/cxxapi.h>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace miniqc {

struct RHFOptions {
    int max_iter = 128;
    double e_conv = 1.0e-10;
    double d_conv = 1.0e-8;
    bool use_diis = true;
    int diis_start = 2;
    std::size_t diis_max_vec = 8;
    bool verbose = true;
};

struct RHFResult {
    bool converged = false;
    double energy_electronic = 0.0;
    double energy_total = 0.0;
    double energy_nuclear = 0.0;
    Eigen::MatrixXd C;
    Eigen::VectorXd eps;
    Eigen::MatrixXd D;
    Eigen::MatrixXd F;
    int iterations = 0;
};

inline RHFResult rhf_closed_shell(const libint2::BasisSet& basis,
                                  const Eigen::MatrixXd& S,
                                  const Eigen::MatrixXd& Hcore,
                                  int nelec,
                                  const Molecule& mol,
                                  const RHFOptions& options = RHFOptions{},
                                  const Eigen::MatrixXd* C_guess = nullptr) {
    if (nelec % 2 != 0) throw std::runtime_error("closed-shell RHF needs even electron count");

    const int nocc = nelec / 2;
    if (nocc <= 0 || nocc > static_cast<int>(basis.nbf())) {
        throw std::runtime_error("invalid occupied orbital count");
    }

    const int nbf = static_cast<int>(basis.nbf());
    const Eigen::MatrixXd X = symmetric_orthogonalization(S);

    Eigen::MatrixXd D;
    const bool have_valid_C_guess =
        C_guess != nullptr &&
        C_guess->rows() == nbf &&
        C_guess->cols() >= nocc &&
        C_guess->allFinite();

    if (have_valid_C_guess) {
        D = build_density(*C_guess, nocc);
    } else {
        Eigen::VectorXd eps0;
        Eigen::MatrixXd C0;
        diagonalize_fock(Hcore, X, eps0, C0);
        D = build_density(C0, nocc);
    }

    double E_old = 0.0;
    miniqc::scf::DIISManager diis(options.diis_max_vec);
    RHFResult result;
    result.energy_nuclear = nuclear_repulsion_energy(mol);

    for (int iter = 1; iter <= options.max_iter; ++iter) {
        const Eigen::MatrixXd G = build_g_direct(basis, D);
        const Eigen::MatrixXd F_raw = Hcore + G;

        const Eigen::MatrixXd err = miniqc::scf::DIISManager::error_matrix(F_raw, D, S, X);
        const double diis_err = err.norm();

        Eigen::MatrixXd F_diag = F_raw;
        if (options.use_diis) {
            diis.add(F_raw, err);
            if (iter >= options.diis_start && diis.size() >= 2) {
                F_diag = diis.extrapolate();
            }
        }

        Eigen::VectorXd eps;
        Eigen::MatrixXd C;
        diagonalize_fock(F_diag, X, eps, C);
        const Eigen::MatrixXd D_new = build_density(C, nocc);

        // Use the physical Fock F_raw = H + G[D] for the reported electronic energy.
        // F_diag may be a DIIS-extrapolated Fock and is only used to generate orbitals.
        const double E_elec = rhf_electronic_energy(D, Hcore, F_raw);
        const double dE = E_elec - E_old;
        const double rmsD = (D_new - D).norm() / static_cast<double>(nbf);

        if (options.verbose) {
            std::cout << "iter " << std::setw(3) << iter
                      << "  E_elec = " << std::setw(18) << std::setprecision(12) << E_elec
                      << "  dE = " << std::setw(13) << std::scientific << dE
                      << "  rmsD = " << rmsD
                      << "  diis = " << diis_err << std::fixed << "\n";
        }

        result.converged = false;
        result.energy_electronic = E_elec;
        result.energy_total = E_elec + result.energy_nuclear;
        result.C = C;
        result.eps = eps;
        result.D = D_new;
        result.F = F_raw;
        result.iterations = iter;

        if (iter > 1 && std::abs(dE) < options.e_conv && rmsD < options.d_conv) {
            const Eigen::MatrixXd G_final = build_g_direct(basis, D_new);
            const Eigen::MatrixXd F_final = Hcore + G_final;
            const double E_final = rhf_electronic_energy(D_new, Hcore, F_final);

            Eigen::VectorXd eps_final;
            Eigen::MatrixXd C_final;
            diagonalize_fock(F_final, X, eps_final, C_final);

            result.converged = true;
            result.energy_electronic = E_final;
            result.energy_total = E_final + result.energy_nuclear;
            result.C = C_final;
            result.eps = eps_final;
            result.D = D_new;
            result.F = F_final;
            result.iterations = iter;
            return result;
        }

        D = D_new;
        E_old = E_elec;
    }

    throw std::runtime_error("RHF did not converge");
}

inline RHFResult rhf_closed_shell(const libint2::BasisSet& basis,
                                  const Eigen::MatrixXd& S,
                                  const Eigen::MatrixXd& Hcore,
                                  int nelec,
                                  const Molecule& mol,
                                  int max_iter,
                                  double e_conv,
                                  double d_conv,
                                  bool use_diis,
                                  int diis_start,
                                  std::size_t diis_max_vec,
                                  bool verbose,
                                  const Eigen::MatrixXd* C_guess = nullptr) {
    RHFOptions options;
    options.max_iter = max_iter;
    options.e_conv = e_conv;
    options.d_conv = d_conv;
    options.use_diis = use_diis;
    options.diis_start = diis_start;
    options.diis_max_vec = diis_max_vec;
    options.verbose = verbose;
    return rhf_closed_shell(basis, S, Hcore, nelec, mol, options, C_guess);
}

inline RHFResult run_rhf_closed_shell(const Molecule& mol,
                                      const std::string& basis_name,
                                      const RHFOptions& options = RHFOptions{},
                                      const Eigen::MatrixXd* C_guess = nullptr,
                                      Eigen::MatrixXd* C_out = nullptr) {
    if (mol.multiplicity != 1) {
        throw std::runtime_error("closed-shell RHF requires multiplicity 1");
    }

    const int nelec = electron_count(mol);
    if (nelec <= 0) throw std::runtime_error("non-positive electron count");
    if (nelec % 2 != 0) throw std::runtime_error("closed-shell RHF needs an even electron count");

    libint2::BasisSet basis(basis_name, mol.atoms, true);
    basis.set_pure(false);

    const OneBodyIntegrals one = build_one_body_integrals(basis, mol.atoms);
    RHFResult rhf = rhf_closed_shell(basis,
                                     one.S,
                                     one.Hcore,
                                     nelec,
                                     mol,
                                     options,
                                     C_guess);

    if (C_out != nullptr) {
        *C_out = rhf.C;
    }

    return rhf;
}

inline double rhf_total_energy_quiet(const Molecule& mol,
                                     const std::string& basis_name,
                                     int max_iter = 128,
                                     const Eigen::MatrixXd* C_guess = nullptr,
                                     Eigen::MatrixXd* C_out = nullptr) {
    RHFOptions options;
    options.max_iter = max_iter;
    options.verbose = false;

    const RHFResult rhf = run_rhf_closed_shell(mol, basis_name, options, C_guess, C_out);
    return rhf.energy_total;
}

}  // namespace miniqc
