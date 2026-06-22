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

// Historical name kept to minimize changes in main.cpp.
// The enum now includes both LDA and GGA choices.
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

struct RKSLDAOptions {
    int max_iter = 128;
    double e_conv = 1.0e-10;
    double d_conv = 1.0e-8;

    // Density damping. Usually keep this at 0.0 when DIIS is enabled.
    // If SCF oscillates badly, use a small value such as 0.1--0.3
    // and/or delay DIIS by increasing diis_start.
    double density_mixing = 0.0;

    // Pulay DIIS for the Kohn-Sham Fock matrix.
    bool use_diis = true;
    int diis_start = 2;
    std::size_t diis_max_vecs = 8;

    // XC functional used in the RKS driver.
    LDAFunctional functional = LDAFunctional::SlaterX;

    bool verbose = true;
};

struct RKSLDAResult {
    bool converged = false;
    int niter = 0;

    double E_total = 0.0;
    double E_elec = 0.0;
    double E_x = 0.0;
    double E_c = 0.0;
    double E_xc = 0.0;
    double E_nuc = 0.0;
    double Ne_grid = 0.0;

    Eigen::VectorXd eps;
    Eigen::MatrixXd C;
    Eigen::MatrixXd D;
    Eigen::MatrixXd F;
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

// Build hand-written LDA Slater exchange-only energy and Vx matrix.
LDAExchangeResult
build_lda_exchange_only_sp(const libint2::BasisSet& basis,
                           const Eigen::MatrixXd& D,
                           const std::vector<GridPoint>& grid,
                           double rho_cutoff = 1.0e-14);

// Build LDA exchange-correlation through Libxc.
LDAExchangeResult
build_lda_libxc_sp(const libint2::BasisSet& basis,
                   const Eigen::MatrixXd& D,
                   const std::vector<GridPoint>& grid,
                   LDAFunctional functional,
                   double rho_cutoff = 1.0e-14);

// Build GGA exchange-correlation through Libxc.
// Currently supports LDAFunctional::Libxc_GGA_PBE.
LDAExchangeResult
build_gga_libxc_sp(const libint2::BasisSet& basis,
                   const Eigen::MatrixXd& D,
                   const std::vector<GridPoint>& grid,
                   LDAFunctional functional,
                   double rho_cutoff = 1.0e-14);

// Uniform entry point used by the SCF driver. SlaterX uses the hand-written
// educational implementation; Libxc_* choices use Libxc.
LDAExchangeResult
build_lda_xc_sp(const libint2::BasisSet& basis,
                const Eigen::MatrixXd& D,
                const std::vector<GridPoint>& grid,
                LDAFunctional functional,
                double rho_cutoff = 1.0e-14);

// Coulomb matrix only:
//   J_mn = sum_ls D_ls (mn|ls)
Eigen::MatrixXd
build_j_direct(const libint2::BasisSet& basis,
               const Eigen::MatrixXd& D);

// RKS SCF driver.
// The historical function name is kept, but it now also supports Libxc GGA PBE.
RKSLDAResult
run_rks_lda_exchange_only_sp(const libint2::BasisSet& basis,
                             const Eigen::MatrixXd& S,
                             const Eigen::MatrixXd& Hcore,
                             int nelec,
                             double Enuc,
                             const std::vector<GridPoint>& grid,
                             const RKSLDAOptions& options = {});

double trace_product(const Eigen::MatrixXd& A,
                     const Eigen::MatrixXd& B);

} // namespace dft

#endif // ABINIT_STUDY_DFT_LDA_HPP
