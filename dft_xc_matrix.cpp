#include "dft_xc_matrix.hpp"

#include "xc_evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace dft {

namespace {

void check_density_shape(const libint2::BasisSet& basis,
                         const Eigen::MatrixXd& D,
                         const char* caller) {
    const int nbf = static_cast<int>(basis.nbf());
    if (D.rows() != nbf || D.cols() != nbf) {
        throw std::runtime_error(std::string(caller) + ": density matrix dimension mismatch");
    }
}

void accumulate_lda_point(const Eigen::VectorXd& phi,
                          double weight,
                          double rho,
                          const miniqc::xc::XCOutputPoint& xc,
                          LDAExchangeResult& result) {
    result.Exc += weight * rho * xc.exc;
    result.Vxc.noalias() += weight * xc.vrho * (phi * phi.transpose());
}

void accumulate_gga_point(const AOValuesGrad& ao,
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

}  // namespace

LDAExchangeResult
build_xc_matrix_with_evaluator_sp(const libint2::BasisSet& basis,
                                  const Eigen::MatrixXd& D,
                                  const std::vector<GridPoint>& grid,
                                  const miniqc::xc::XCFunctional& functional,
                                  double rho_cutoff) {
    check_density_shape(basis, D, "build_xc_matrix_with_evaluator_sp");

    if (functional.spin_mode != miniqc::xc::SpinMode::Restricted) {
        throw std::runtime_error(
            "build_xc_matrix_with_evaluator_sp: only restricted XC functionals are implemented");
    }
    if (functional.is_meta_gga()) {
        throw std::runtime_error(
            "build_xc_matrix_with_evaluator_sp: meta-GGA tau/vtau matrix terms are not implemented yet");
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
            const miniqc::xc::XCOutputPoint xc =
                miniqc::xc::evaluate_xc_point(functional, input);

            accumulate_gga_point(ao, den, gp.w, rho, xc, result);

            result.Ne_grid += gp.w * rho;
            result.rho_max = std::max(result.rho_max, rho);
            result.sigma_max = std::max(result.sigma_max, den.sigma);
            result.npoints_used += 1;
        } else {
            const Eigen::VectorXd phi = eval_ao_values(basis, gp.r);
            const Eigen::VectorXd Dphi = D * phi;

            double rho = phi.dot(Dphi);
            if (rho < 0.0 && rho > -1.0e-12) rho = 0.0;
            if (rho <= rho_cutoff) continue;

            miniqc::xc::XCInputPoint input;
            input.rho = rho;
            const miniqc::xc::XCOutputPoint xc =
                miniqc::xc::evaluate_xc_point(functional, input);

            accumulate_lda_point(phi, gp.w, rho, xc, result);

            result.Ne_grid += gp.w * rho;
            result.rho_max = std::max(result.rho_max, rho);
            result.npoints_used += 1;
        }
    }

    result.Vxc = 0.5 * (result.Vxc + result.Vxc.transpose());

    // This evaluator currently returns only the total semilocal XC contribution.
    // Keep the historical LDAExchangeResult fields populated in a conservative
    // way so existing callers can inspect total Exc/Vxc.
    result.Vx = result.Vxc;
    result.Vc.setZero(nbf, nbf);
    result.Ex = result.Exc;
    result.Ec = 0.0;

    return result;
}

}  // namespace dft
