#pragma once

#include "dft_lda.hpp"
#include "xc_functional.hpp"

#include <Eigen/Dense>
#include <libint2.hpp>

#include <cstddef>
#include <vector>

namespace miniqc::rks {

struct RKSXCOptions {
    int max_iter = 128;
    double e_conv = 1.0e-10;
    double d_conv = 1.0e-8;

    bool use_diis = true;
    int diis_start = 2;
    std::size_t diis_max_vecs = 8;

    // Fraction of the previous density retained after diagonalization.
    // Keep at 0.0 when DIIS is stable.
    double density_mixing = 0.0;

    double rho_cutoff = 1.0e-14;
    bool verbose = true;
};

struct RKSXCResult {
    bool converged = false;
    int niter = 0;

    double E_total = 0.0;
    double E_elec = 0.0;
    double E_one = 0.0;
    double E_coulomb = 0.0;
    double E_xc = 0.0;
    double E_exact_exchange = 0.0;
    double E_nuc = 0.0;
    double Ne_grid = 0.0;

    Eigen::VectorXd eps;
    Eigen::MatrixXd C;
    Eigen::MatrixXd D;
    Eigen::MatrixXd F;
};

// Total-Exc-oriented restricted Kohn-Sham SCF driver.
//
// This driver uses XCFunctional + XCEvaluator-backed Vxc construction and is
// intended to replace the old LDAExchangeResult Ex/Ec-oriented path for new
// DFT work.  It supports closed-shell restricted calculations with pure LDA/GGA
// and full-range hybrid GGA functionals whose semilocal part can be evaluated
// by the current XCEvaluator.
RKSXCResult run_rks_xc(const libint2::BasisSet& basis,
                       const Eigen::MatrixXd& S,
                       const Eigen::MatrixXd& Hcore,
                       int nelec,
                       double Enuc,
                       const std::vector<dft::GridPoint>& grid,
                       const xc::XCFunctional& functional,
                       const RKSXCOptions& options = {});

}  // namespace miniqc::rks
