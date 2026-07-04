#pragma once

#include "xc_functional.hpp"

#include <cstddef>
#include <vector>

namespace miniqc::xc {

struct XCInputPoint {
    double rho = 0.0;
    double sigma = 0.0;
    double laplacian = 0.0;
    double tau = 0.0;
};

struct XCOutputPoint {
    // Libxc returns eps_xc as energy per particle.  The total XC energy
    // contribution at a grid point is w * rho * exc.
    double exc = 0.0;

    // First derivatives of eps/rho convention as returned by Libxc.
    double vrho = 0.0;
    double vsigma = 0.0;
    double vlaplacian = 0.0;
    double vtau = 0.0;
};

struct XCInputBlock {
    std::vector<double> rho;
    std::vector<double> sigma;
    std::vector<double> laplacian;
    std::vector<double> tau;

    std::size_t size() const { return rho.size(); }
};

struct XCOutputBlock {
    std::vector<double> exc;
    std::vector<double> vrho;
    std::vector<double> vsigma;
    std::vector<double> vlaplacian;
    std::vector<double> vtau;

    void resize(std::size_t n);
};

// Evaluate a restricted/unpolarized XC functional at one point.
// This is a scalar convenience wrapper around evaluate_xc_block().
XCOutputPoint evaluate_xc_point(const XCFunctional& functional,
                                const XCInputPoint& input);

// Evaluate a restricted/unpolarized XC functional on a block of grid points.
//
// Current scope:
//   - LDA: needs rho.
//   - GGA / HybridGGA: needs rho and sigma.
//
// Meta-GGA support is intentionally not implemented here yet; M06-2X is
// described by XCFunctional but still requires tau and vtau support in the
// future evaluator/RKS matrix builder.
XCOutputBlock evaluate_xc_block(const XCFunctional& functional,
                                const XCInputBlock& input);

}  // namespace miniqc::xc
