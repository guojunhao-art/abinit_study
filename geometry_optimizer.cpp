#include "geometry_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace miniqc {

namespace {

double max_abs_component(const Eigen::VectorXd& v) {
    if (v.size() == 0) return 0.0;
    return v.cwiseAbs().maxCoeff();
}

bool is_finite_vector(const Eigen::VectorXd& v) {
    for (int i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v(i))) return false;
    }
    return true;
}

Eigen::VectorXd limit_step(const Eigen::VectorXd& step, double max_step) {
    if (max_step <= 0.0) return step;
    const double max_disp = max_abs_component(step);
    if (max_disp <= max_step || max_disp == 0.0) return step;
    return step * (max_step / max_disp);
}

}  // namespace

Eigen::VectorXd finite_difference_gradient(const Eigen::VectorXd& x,
                                           const EnergyFunction& energy,
                                           double h,
                                           const Eigen::MatrixXd* C_guess) {
    if (h <= 0.0) throw std::runtime_error("finite-difference step must be positive");

    Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
    Eigen::VectorXd xp = x;
    Eigen::VectorXd xm = x;

    for (int i = 0; i < x.size(); ++i) {
        xp(i) += h;
        xm(i) -= h;

        // Use the same current-geometry C as the initial guess for both displaced
        // calculations. Do not let +h overwrite the guess used by -h.
        const double ep = energy(xp, C_guess, nullptr);
        const double em = energy(xm, C_guess, nullptr);
        if (!std::isfinite(ep) || !std::isfinite(em)) {
            throw std::runtime_error("non-finite energy during numerical-gradient evaluation");
        }

        grad(i) = (ep - em) / (2.0 * h);

        xp(i) = x(i);
        xm(i) = x(i);
    }

    return grad;
}

GeometryResult optimize_bfgs(const Eigen::VectorXd& x0,
                             const EnergyFunction& energy,
                             const GeometryOptions& options) {
    if (x0.size() == 0) throw std::runtime_error("empty coordinate vector");
    if (!is_finite_vector(x0)) throw std::runtime_error("non-finite initial coordinates");

    GeometryResult result;
    Eigen::VectorXd x = x0;

    // Converged MO coefficients at the current geometry. Empty means no guess yet.
    Eigen::MatrixXd C_current;

    double e = energy(x, nullptr, &C_current);
    if (!std::isfinite(e)) throw std::runtime_error("non-finite initial energy");

    Eigen::VectorXd g = finite_difference_gradient(
        x, energy, options.finite_difference_step,
        C_current.size() > 0 ? &C_current : nullptr);
    if (!is_finite_vector(g)) throw std::runtime_error("non-finite initial gradient");

    const int n = static_cast<int>(x.size());
    Eigen::MatrixXd Hinv = Eigen::MatrixXd::Identity(n, n);

    if (options.verbose) {
        std::cout << "\n=== Cartesian BFGS geometry optimization ===\n";
        std::cout << "finite-difference step = " << options.finite_difference_step << " bohr\n";
        std::cout << "max Cartesian step     = " << options.max_step << " bohr\n";
        std::cout << "\n";
        std::cout << std::setw(5) << "step"
                  << std::setw(20) << "E_total/Ha"
                  << std::setw(16) << "dE/Ha"
                  << std::setw(16) << "max|grad|"
                  << std::setw(16) << "rms_grad"
                  << std::setw(16) << "max_disp"
                  << "\n";
    }

    double e_prev = std::numeric_limits<double>::quiet_NaN();

    for (int iter = 0; iter <= options.max_steps; ++iter) {
        const double max_grad = max_abs_component(g);
        const double rms_grad = g.norm() / std::sqrt(static_cast<double>(n));
        const double dE_print = (iter == 0 || !std::isfinite(e_prev)) ? 0.0 : e - e_prev;

        if (options.verbose) {
            std::cout << std::setw(5) << iter
                      << std::setw(20) << std::fixed << std::setprecision(12) << e
                      << std::setw(16) << std::scientific << dE_print
                      << std::setw(16) << max_grad
                      << std::setw(16) << rms_grad
                      << std::setw(16) << 0.0
                      << std::fixed << "\n";
        }

        const bool force_converged =
            max_grad < options.max_force_conv && rms_grad < options.rms_force_conv;
        const bool energy_converged =
            iter > 0 && std::abs(e - e_prev) < options.energy_conv;

        if (force_converged && energy_converged) {
            result.x = x;
            result.gradient = g;
            result.energy = e;
            result.iterations = iter;
            result.converged = true;
            return result;
        }

        if (iter == options.max_steps) break;

        Eigen::VectorXd p = -Hinv * g;
        if (!is_finite_vector(p) || p.dot(g) >= 0.0) {
            // Reset if BFGS inverse Hessian has become non-descent.
            Hinv.setIdentity();
            p = -g;
        }

        p = limit_step(p, options.max_step);
        const double g_dot_p = g.dot(p);
        if (g_dot_p >= 0.0) {
            p = limit_step(-g, options.max_step);
        }

        // Backtracking Armijo line search. Since p was already step-limited,
        // alpha=1 is usually the desired update.
        double alpha = 1.0;
        const double c1 = 1.0e-4;
        Eigen::VectorXd x_trial;
        Eigen::MatrixXd C_trial;
        double e_trial = std::numeric_limits<double>::quiet_NaN();
        bool accepted = false;

        for (int ls = 0; ls < 20; ++ls) {
            x_trial = x + alpha * p;
            Eigen::MatrixXd C_trial_candidate;
            e_trial = energy(x_trial,
                             C_current.size() > 0 ? &C_current : nullptr,
                             &C_trial_candidate);
            if (std::isfinite(e_trial) && e_trial <= e + c1 * alpha * g.dot(p)) {
                C_trial = std::move(C_trial_candidate);
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!accepted) {
            // Conservative fallback: small steepest-descent step.
            Eigen::VectorXd sd = limit_step(-g, 0.05 * options.max_step);
            x_trial = x + sd;
            C_trial.resize(0, 0);
            e_trial = energy(x_trial,
                             C_current.size() > 0 ? &C_current : nullptr,
                             &C_trial);
            if (!std::isfinite(e_trial) || e_trial > e) {
                throw std::runtime_error("geometry line search failed to find a lower-energy step");
            }
            p = sd;
            alpha = 1.0;
            Hinv.setIdentity();
        }

        Eigen::VectorXd g_trial = finite_difference_gradient(
            x_trial, energy, options.finite_difference_step,
            C_trial.size() > 0 ? &C_trial : nullptr);
        if (!is_finite_vector(g_trial)) {
            throw std::runtime_error("non-finite gradient after geometry step");
        }

        const Eigen::VectorXd s = x_trial - x;
        const Eigen::VectorXd y = g_trial - g;
        const double ys = y.dot(s);

        if (ys > 1.0e-10) {
            const double rho = 1.0 / ys;
            const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
            Hinv = (I - rho * s * y.transpose()) * Hinv *
                   (I - rho * y * s.transpose()) + rho * s * s.transpose();
            Hinv = 0.5 * (Hinv + Hinv.transpose());
        } else {
            // Curvature condition failed; reset to avoid a bad quasi-Newton metric.
            Hinv.setIdentity();
        }

        e_prev = e;
        x = x_trial;
        e = e_trial;
        g = g_trial;
        C_current = std::move(C_trial);

        if (options.verbose) {
            const double max_disp = max_abs_component(s);
            std::cout << "      accepted step: alpha = " << std::scientific << alpha
                      << ", max_disp = " << max_disp << std::fixed << "\n";
        }
    }

    result.x = x;
    result.gradient = g;
    result.energy = e;
    result.iterations = options.max_steps;
    result.converged = false;
    return result;
}

}  // namespace miniqc
