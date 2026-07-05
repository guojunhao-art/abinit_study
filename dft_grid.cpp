#include "dft_grid.hpp"
#include "lebedev_grid_high_order.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace dft {

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

Eigen::Vector3d atom_position(const libint2::Atom& atom) {
    return Eigen::Vector3d{atom.x, atom.y, atom.z};
}

double clamp_value(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

double becke_p(double x) {
    return 0.5 * (3.0 * x - x * x * x);
}

double becke_smooth(double mu, int order) {
    double x = clamp_value(mu, -1.0, 1.0);

    for (int i = 0; i < order; ++i) {
        x = becke_p(x);
    }

    return x;
}

} // namespace


std::vector<AngularGridPoint> make_angular_grid(int angular_grid) {
    const auto lebedev_points = lebedev::make_lebedev_grid_extended(
        angular_grid,
        true  // weights sum to 4*pi
    );

    std::vector<AngularGridPoint> grid;
    grid.reserve(lebedev_points.size());

    for (const auto& p : lebedev_points) {
        AngularGridPoint gp;
        gp.u = p.u;
        gp.w = p.w;
        grid.push_back(gp);
    }

    return grid;
}


std::vector<RadialGridPoint>
make_radial_grid(const AtomCenteredGridOptions& options) {
    if (options.n_radial == 0) {
        throw std::runtime_error("make_radial_grid: n_radial must be positive");
    }
    if (options.r_max <= 0.0) {
        throw std::runtime_error("make_radial_grid: r_max must be positive");
    }

    std::vector<RadialGridPoint> radial;
    radial.reserve(options.n_radial);

    const double n = static_cast<double>(options.n_radial);

    if (options.radial_type == RadialGridType::Uniform) {
        // Midpoint quadrature on r in [0, r_max]:
        //
        //   r_i = (i + 1/2) dr
        //   w_i^rad = dr
        //
        // The full spherical volume weight is formed later:
        //   w_raw = w_i^rad * r_i^2 * w_k^ang
        const double dr = options.r_max / n;

        for (std::size_t i = 0; i < options.n_radial; ++i) {
            RadialGridPoint gp;
            gp.r = (static_cast<double>(i) + 0.5) * dr;
            gp.w_rad = dr;
            radial.push_back(gp);
        }

        return radial;
    }

    if (options.radial_type == RadialGridType::PowerMapped) {
        // Midpoint quadrature on t in [0,1], with:
        //
        //   r = r_max * t^q
        //
        // Hence:
        //
        //   dr/dt = r_max * q * t^(q - 1)
        //
        // and:
        //
        //   w_i^rad = (dr/dt)|_{t_i} * Delta t
        //             = r_max * q * t_i^(q - 1) / n_radial
        //
        // q > 1 clusters radial points near the nucleus.
        const double q = options.radial_power;
        if (q <= 0.0) {
            throw std::runtime_error("make_radial_grid: radial_power must be positive");
        }

        for (std::size_t i = 0; i < options.n_radial; ++i) {
            const double t = (static_cast<double>(i) + 0.5) / n;
            const double r = options.r_max * std::pow(t, q);
            const double drdt = options.r_max * q * std::pow(t, q - 1.0);

            RadialGridPoint gp;
            gp.r = r;
            gp.w_rad = drdt / n;
            radial.push_back(gp);
        }

        return radial;
    }

    throw std::runtime_error("make_radial_grid: unknown radial_type");
}


double distance_partition_weight(const std::vector<libint2::Atom>& atoms,
                                 std::size_t atom_index,
                                 const Eigen::Vector3d& r,
                                 double eps,
                                 double power) {
    if (atoms.empty()) {
        throw std::runtime_error("distance_partition_weight called with no atoms");
    }

    if (atom_index >= atoms.size()) {
        throw std::runtime_error("atom_index out of range in distance_partition_weight");
    }

    if (!(eps > 0.0)) {
        throw std::runtime_error("partition eps must be positive");
    }

    if (!(power > 0.0)) {
        throw std::runtime_error("partition power must be positive");
    }

    double q_sum = 0.0;
    double q_A = 0.0;

    for (std::size_t B = 0; B < atoms.size(); ++B) {
        const Eigen::Vector3d RB = atom_position(atoms[B]);
        const double d = (r - RB).norm();

        const double q = 1.0 / std::pow(d + eps, power);

        if (!std::isfinite(q)) {
            throw std::runtime_error("non-finite distance partition q value");
        }

        q_sum += q;

        if (B == atom_index) {
            q_A = q;
        }
    }

    if (!(q_sum > 0.0) || !std::isfinite(q_sum)) {
        throw std::runtime_error("invalid distance partition q_sum");
    }

    return q_A / q_sum;
}

double becke_partition_weight(const std::vector<libint2::Atom>& atoms,
                              std::size_t atom_index,
                              const Eigen::Vector3d& r,
                              int smooth_order) {
    if (atoms.empty()) {
        throw std::runtime_error("becke_partition_weight called with no atoms");
    }

    if (atom_index >= atoms.size()) {
        throw std::runtime_error("atom_index out of range in becke_partition_weight");
    }

    if (smooth_order < 1) {
        throw std::runtime_error("Becke smooth_order must be >= 1");
    }

    const std::size_t natom = atoms.size();

    std::vector<double> dist(natom, 0.0);
    for (std::size_t A = 0; A < natom; ++A) {
        const Eigen::Vector3d RA = atom_position(atoms[A]);
        dist[A] = (r - RA).norm();
    }

    std::vector<double> cell(natom, 1.0);

    for (std::size_t A = 0; A < natom; ++A) {
        const Eigen::Vector3d RA = atom_position(atoms[A]);

        double pA = 1.0;

        for (std::size_t B = 0; B < natom; ++B) {
            if (B == A) continue;

            const Eigen::Vector3d RB = atom_position(atoms[B]);
            const double RAB = (RA - RB).norm();

            if (RAB < 1.0e-12) {
                throw std::runtime_error("two atoms are too close in Becke partition");
            }

            const double rA = dist[A];
            const double rB = dist[B];
            const double mu = (rA - rB) / RAB;
            const double s = becke_smooth(mu, smooth_order);
            pA *= 0.5 * (1.0 - s);
        }

        cell[A] = pA;
    }

    double denom = 0.0;
    for (double p : cell) {
        denom += p;
    }

    if (!(denom > 0.0) || !std::isfinite(denom)) {
        // The symmetric construction can underflow in the far field for large
        // molecules.  Fall back to a smooth distance partition so every grid
        // point still contributes exactly once across atoms.
        return distance_partition_weight(atoms, atom_index, r, 1.0e-12, 2.0);
    }

    return cell[atom_index] / denom;
}


std::vector<GridPoint>
make_atom_centered_grid(const std::vector<libint2::Atom>& atoms,
                        const AtomCenteredGridOptions& options) {
    if (atoms.empty()) {
        throw std::runtime_error("make_atom_centered_grid: empty atom list");
    }
    if (!lebedev::is_supported_extended(options.angular_grid)) {
        throw std::runtime_error(
            "make_atom_centered_grid: unsupported Lebedev angular grid size " +
            std::to_string(options.angular_grid)
        );
    }

    const auto angular = make_angular_grid(options.angular_grid);
    const auto radial = make_radial_grid(options);

    std::vector<GridPoint> grid;
    grid.reserve(atoms.size() * angular.size() * radial.size());

    for (std::size_t A = 0; A < atoms.size(); ++A) {
        const Eigen::Vector3d RA = atom_position(atoms[A]);
        for (const auto& rg : radial) {
            const double r2 = rg.r * rg.r;
            for (const auto& ag : angular) {
                GridPoint gp;
                gp.r = RA + rg.r * ag.u;
                const double w_raw = rg.w_rad * r2 * ag.w;
                const double pA = becke_partition_weight(atoms, A, gp.r, options.becke_smooth_order);
                gp.w = w_raw * pA;
                grid.push_back(gp);
            }
        }
    }

    return grid;
}

} // namespace dft
