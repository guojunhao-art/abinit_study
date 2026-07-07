#include "dispersion.hpp"

#include <libint2/cxxapi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

libint2::Atom atom(int Z, double x, double y, double z) {
    libint2::Atom a;
    a.atomic_number = Z;
    a.x = x;
    a.y = y;
    a.z = z;
    return a;
}

void check_hydrogen_pair_against_manual_formula() {
    const double r = 10.0;
    const double s6 = 1.0;
    const double damping_d = 20.0;

    const std::vector<libint2::Atom> atoms = {
        atom(1, 0.0, 0.0, 0.0),
        atom(1, r, 0.0, 0.0),
    };

    const double e = miniqc::dispersion::compute_d2_dispersion_energy(atoms, s6, damping_d);

    const auto h = miniqc::dispersion::d2_parameters_for_atomic_number(1);
    const double c6 = h.c6_j_nm6_mol * miniqc::dispersion::d2_c6_conversion_to_atomic_units();
    const double r0 = 2.0 * h.r0_angstrom * 1.88972612456506;
    const double fdamp = 1.0 / (1.0 + std::exp(-damping_d * (r / r0 - 1.0)));
    const double expected = -s6 * c6 * fdamp / std::pow(r, 6);

    require(std::abs(e - expected) < 1.0e-18, "D2 H-H pair energy mismatch");
    require(e < 0.0, "D2 dispersion energy should be attractive");
}

void check_pair_symmetry() {
    const std::vector<libint2::Atom> co = {
        atom(6, 0.0, 0.0, 0.0),
        atom(8, 7.0, 0.0, 0.0),
    };
    const std::vector<libint2::Atom> oc = {
        atom(8, 7.0, 0.0, 0.0),
        atom(6, 0.0, 0.0, 0.0),
    };

    const double eco = miniqc::dispersion::compute_d2_dispersion_energy(co, 0.75, 20.0);
    const double eoc = miniqc::dispersion::compute_d2_dispersion_energy(oc, 0.75, 20.0);
    require(std::abs(eco - eoc) < 1.0e-18, "D2 energy should be atom-order invariant");
}

void check_dispatch_and_defaults() {
    miniqc::dispersion::DispersionOptions options;
    options.method = miniqc::dispersion::DispersionMethod::D2;
    options.functional = "pbe";

    const std::vector<libint2::Atom> atoms = {
        atom(6, 0.0, 0.0, 0.0),
        atom(6, 8.0, 0.0, 0.0),
    };

    const auto result = miniqc::dispersion::compute_dispersion_energy(atoms, options);
    require(result.method == miniqc::dispersion::DispersionMethod::D2, "D2 method dispatch failed");
    require(std::abs(result.s6 - 0.75) < 1.0e-15, "PBE-D2 default s6 mismatch");
    require(result.energy < 0.0, "D2 dispatch energy should be attractive");
}

void check_missing_parameters_throw() {
    bool threw = false;
    try {
        const std::vector<libint2::Atom> atoms = {
            atom(2, 0.0, 0.0, 0.0),
            atom(2, 8.0, 0.0, 0.0),
        };
        (void)miniqc::dispersion::compute_d2_dispersion_energy(atoms, 1.0, 20.0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "D2 should reject elements without table parameters");
}

}  // namespace

int main() {
    check_hydrogen_pair_against_manual_formula();
    check_pair_symmetry();
    check_dispatch_and_defaults();
    check_missing_parameters_throw();

    std::cout << "D2 dispersion test passed\n";
    return 0;
}
