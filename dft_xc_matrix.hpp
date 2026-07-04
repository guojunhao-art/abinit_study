#pragma once

#include "dft_lda.hpp"
#include "xc_functional.hpp"

#include <Eigen/Dense>
#include <libint2.hpp>

#include <vector>

namespace dft {

// Build the semilocal XC energy and Vxc matrix using the miniqc::xc evaluator.
//
// This function is intentionally separate from the historical build_lda_xc_sp()
// path.  It is the bridge between the new XCFunctional/XCEvaluator layers and
// the existing AO-grid matrix assembly code.
LDAExchangeResult
build_xc_matrix_with_evaluator_sp(const libint2::BasisSet& basis,
                                  const Eigen::MatrixXd& D,
                                  const std::vector<GridPoint>& grid,
                                  const miniqc::xc::XCFunctional& functional,
                                  double rho_cutoff = 1.0e-14);

}  // namespace dft
