#ifndef ABINIT_STUDY_DFT_LDA_HPP
#define ABINIT_STUDY_DFT_LDA_HPP

#include <Eigen/Dense>
#include <libint2.hpp>

#include <cstddef>
#include <vector>

namespace dft {

struct GridPoint {
    Eigen::Vector3d r;  // bohr
    double w = 0.0;     // bohr^3, may be negative for some Lebedev rules
};

struct UniformGridOptions {
    double spacing = 0.35;   // bohr
    double padding = 6.0;    // bohr
    std::size_t max_points = 20000000;
};

// Historical enum used by legacy XC matrix helper tests and compatibility
// wrappers.  New user-facing DFT calculations should use miniqc::xc::XCFunctional
// and miniqc::rks::run_rks_xc().
enum class LDAFunctional {
    // Hand-written Slater exchange, useful for learning and debugging.
    SlaterX,

    // Libxc LDA exchange only: XC_LDA_X.
    Libxc_LDA_X,

    // Libxc Slater exchange + Perdew-Zunger 1981 correlation:
    // XC_LDA_X + XC_LDA_C_PZ.
    Libxc_LDA_X_PZ81C,

    // Libxc PBE GGA exchange-correlation:
    // XC_GGA_X_PBE + XC_GGA_C_PBE.
    Libxc_GGA_PBE
};

struct LDAExchangeResult {
    Eigen::MatrixXd Vx;
    Eigen::MatrixXd Vc;
    Eigen::MatrixXd Vxc;

    double Ex = 0.0;
    double Ec = 0.0;
    double Exc = 0.0;

    double Ne_grid = 0.0;
    double rho_max = 0.0;
    double sigma_max = 0.0;
    std::size_t npoints_used = 0;
};

struct AOValuesGrad {
    Eigen::VectorXd phi;
    Eigen::VectorXd dphidx;
    Eigen::VectorXd dphidy;
    Eigen::VectorXd dphidz;
};

struct DensityGradPoint {
    double rho = 0.0;
    Eigen::Vector3d grad_rho = Eigen::Vector3d::Zero();
    double sigma = 0.0;  // |grad rho|^2 for unpolarized RKS GGA
};

// Simple uniform Cartesian grid for the first educational implementation.
// Coordinates must already be in bohr.
std::vector<GridPoint>
make_uniform_grid(const std::vector<libint2::Atom>& atoms,
                  const UniformGridOptions& options);

// Evaluate AO values at one point.
// This educational evaluator supports Cartesian Gaussian shells of arbitrary
// angular momentum l, using Libint's Cartesian ordering:
//   l=0: s
//   l=1: x, y, z
//   l=2: xx, xy, xz, yy, yz, zz
//   l=3: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
// and so on.
//
// Important: this routine intentionally supports Cartesian shells only.
// If a Libint shell is pure/spherical, it throws, because the real spherical
// harmonic normalization/order must exactly match the integral basis.
Eigen::VectorXd
eval_ao_values(const libint2::BasisSet& basis,
               const Eigen::Vector3d& r);

// Backward-compatible name used by older DFT code.
Eigen::VectorXd
eval_ao_values_sp_only(const libint2::BasisSet& basis,
                       const Eigen::Vector3d& r);

// AO values and Cartesian derivatives at one point.
AOValuesGrad
eval_ao_values_grad(const libint2::BasisSet& basis,
                    const Eigen::Vector3d& r);

// rho, grad rho, and sigma = |grad rho|^2 for a closed-shell total density D.
DensityGradPoint
eval_density_gradient(const Eigen::MatrixXd& D,
                      const AOValuesGrad& ao);

DensityGradPoint
eval_density_gradient(const libint2::BasisSet& basis,
                      const Eigen::MatrixXd& D,
                      const Eigen::Vector3d& r);

// Debug helpers: compare analytic gradients against central finite difference.
double
check_ao_gradient_finite_difference(const libint2::BasisSet& basis,
                                    const Eigen::Vector3d& r,
                                    double h = 1.0e-5,
                                    bool verbose = true);

double
check_density_gradient_finite_difference(const libint2::BasisSet& basis,
                                         const Eigen::MatrixXd& D,
                                         const Eigen::Vector3d& r,
                                         double h = 1.0e-5,
                                         bool verbose = true);

// Legacy semilocal XC matrix helpers kept temporarily for regression tests and
// for comparison against the new XCEvaluator-backed matrix builder.  They no
// longer own an SCF driver.
LDAExchangeResult
build_lda_exchange_only_sp(const libint2::BasisSet& basis,
                           const Eigen::MatrixXd& D,
                           const std::vector<GridPoint>& grid,
                           double rho_cutoff = 1.0e-14);

LDAExchangeResult
build_lda_libxc_sp(const libint2::BasisSet& basis,
                   const Eigen::MatrixXd& D,
                   const std::vector<GridPoint>& grid,
                   LDAFunctional functional,
                   double rho_cutoff = 1.0e-14);

LDAExchangeResult
build_gga_libxc_sp(const libint2::BasisSet& basis,
                   const Eigen::MatrixXd& D,
                   const std::vector<GridPoint>& grid,
                   LDAFunctional functional,
                   double rho_cutoff = 1.0e-14);

LDAExchangeResult
build_lda_xc_sp(const libint2::BasisSet& basis,
                const Eigen::MatrixXd& D,
                const std::vector<GridPoint>& grid,
                LDAFunctional functional,
                double rho_cutoff = 1.0e-14);

// Legacy Coulomb helper in namespace dft.  New code should prefer
// miniqc::build_j_direct() from two_body_fock.hpp.
Eigen::MatrixXd
build_j_direct(const libint2::BasisSet& basis,
               const Eigen::MatrixXd& D);

double trace_product(const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& B);

} // namespace dft

#endif // ABINIT_STUDY_DFT_LDA_HPP
