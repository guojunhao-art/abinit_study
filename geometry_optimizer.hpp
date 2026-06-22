#pragma once

#include <Eigen/Dense>
#include <functional>
#include <string>

namespace miniqc {

struct GeometryOptions {
    int max_steps = 50;
    double finite_difference_step = 1.0e-3;  // bohr
    double max_step = 0.20;                  // bohr, max Cartesian displacement per step
    double energy_conv = 1.0e-7;             // Hartree
    double max_force_conv = 4.5e-4;          // Hartree/bohr
    double rms_force_conv = 3.0e-4;          // Hartree/bohr
    bool verbose = true;
};

struct GeometryResult {
    Eigen::VectorXd x;
    Eigen::VectorXd gradient;
    double energy = 0.0;
    int iterations = 0;
    bool converged = false;
};

// Energy function with optional SCF warm start.
// C_guess: previous converged MO coefficients used as initial SCF guess; may be nullptr.
// C_out: if non-null, receives converged MO coefficients at this geometry.
using EnergyFunction = std::function<double(const Eigen::VectorXd&,
                                            const Eigen::MatrixXd* C_guess,
                                            Eigen::MatrixXd* C_out)>;

using GradientFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd& x,
                                  const Eigen::MatrixXd* C_guess,
                                  Eigen::MatrixXd* C_out)>;

Eigen::VectorXd finite_difference_gradient(const Eigen::VectorXd& x,
                                           const EnergyFunction& energy,
                                           double h,
                                           const Eigen::MatrixXd* C_guess = nullptr);

GeometryResult optimize_bfgs(const Eigen::VectorXd& x0,
                             const EnergyFunction& energy,
                             const GeometryOptions& options = GeometryOptions{},
                             const GradientFunction& gradient = nullptr);

Eigen::VectorXd
flatten_atom_gradient_rowwise(const Eigen::MatrixXd& grad);


}  // namespace miniqc
