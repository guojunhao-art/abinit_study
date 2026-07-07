#include "dispersion.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <stdexcept>

namespace miniqc::dispersion {

namespace {

constexpr double kAvogadro = 6.02214076e23;
constexpr double kHartreeJ = 4.3597447222071e-18;
constexpr double kBohrPerNanometer = 18.8972612456506;
constexpr double kBohrPerAngstrom = 1.88972612456506;

std::string normalize_key(std::string s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isspace(c) || c == '_' || c == '-' || c == '(' || c == ')' || c == '/') continue;
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

double distance_bohr(const libint2::Atom& a, const libint2::Atom& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double fermi_damping(double r_bohr, double r0_bohr, double damping_d) {
    if (!(r_bohr > 0.0) || !(r0_bohr > 0.0)) {
        throw std::runtime_error("invalid D2 damping distance");
    }
    const double x = -damping_d * (r_bohr / r0_bohr - 1.0);
    if (x > 50.0) return std::exp(-x);       // numerically same as 1/(1+exp(x))
    if (x < -50.0) return 1.0;
    return 1.0 / (1.0 + std::exp(x));
}

}  // namespace

std::string normalize_dispersion_name(const std::string& name) {
    return normalize_key(name);
}

DispersionMethod parse_dispersion_method(const std::string& name) {
    const std::string key = normalize_key(name);
    if (key.empty() || key == "none" || key == "off" || key == "false" || key == "0") {
        return DispersionMethod::None;
    }
    if (key == "d2" || key == "grimme2" || key == "grimmed2" || key == "dftd2") {
        return DispersionMethod::D2;
    }
    throw std::runtime_error("unknown dispersion correction method: " + name);
}

std::string to_string(DispersionMethod method) {
    switch (method) {
        case DispersionMethod::None: return "none";
        case DispersionMethod::D2: return "d2";
    }
    return "unknown";
}

bool has_d2_parameters(int atomic_number) {
    switch (atomic_number) {
        case 1:   // H
        case 6:   // C
        case 7:   // N
        case 8:   // O
        case 9:   // F
        case 15:  // P
        case 16:  // S
        case 17:  // Cl
        case 35:  // Br
        case 53:  // I
            return true;
        default:
            return false;
    }
}

D2AtomParameters d2_parameters_for_atomic_number(int atomic_number) {
    // Grimme D2 atomic C6 values are in J nm^6 mol^-1 and R0 values are in Angstrom.
    switch (atomic_number) {
        case 1:  return {0.14, 1.001};  // H
        case 6:  return {1.75, 1.452};  // C
        case 7:  return {1.23, 1.397};  // N
        case 8:  return {0.70, 1.342};  // O
        case 9:  return {0.75, 1.287};  // F
        case 15: return {7.84, 1.705};  // P
        case 16: return {5.57, 1.683};  // S
        case 17: return {5.07, 1.639};  // Cl
        case 35: return {12.47, 1.749}; // Br
        case 53: return {31.50, 1.892}; // I
        default:
            throw std::runtime_error("missing D2 parameters for atomic number " +
                                     std::to_string(atomic_number));
    }
}

double d2_c6_conversion_to_atomic_units() {
    // Convert J nm^6 mol^-1 to Hartree bohr^6.
    return std::pow(kBohrPerNanometer, 6) / (kAvogadro * kHartreeJ);
}

double d2_default_s6_for_functional(const std::string& functional) {
    const std::string key = normalize_key(functional);
    if (key == "pbe") return 0.75;
    if (key == "blyp") return 1.20;
    if (key == "b3lyp") return 1.05;
    if (key == "pbe0" || key == "pbeh") return 0.60;
    throw std::runtime_error(
        "no built-in D2 s6 parameter for functional " + functional +
        "; set dispersion.s6 explicitly"
    );
}

double compute_d2_dispersion_energy(const std::vector<libint2::Atom>& atoms,
                                    double s6,
                                    double damping_d) {
    if (!std::isfinite(s6)) {
        throw std::runtime_error("D2 s6 scale is not finite");
    }
    if (!(damping_d > 0.0) || !std::isfinite(damping_d)) {
        throw std::runtime_error("D2 damping parameter must be positive and finite");
    }

    const double c6conv = d2_c6_conversion_to_atomic_units();
    double energy = 0.0;

    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const auto pi = d2_parameters_for_atomic_number(atoms[i].atomic_number);
        for (std::size_t j = i + 1; j < atoms.size(); ++j) {
            const auto pj = d2_parameters_for_atomic_number(atoms[j].atomic_number);
            const double r = distance_bohr(atoms[i], atoms[j]);
            if (r < 1.0e-12) {
                throw std::runtime_error("two atoms overlap in D2 dispersion correction");
            }

            const double c6ij = std::sqrt(pi.c6_j_nm6_mol * pj.c6_j_nm6_mol) * c6conv;
            const double r0ij = (pi.r0_angstrom + pj.r0_angstrom) * kBohrPerAngstrom;
            const double damp = fermi_damping(r, r0ij, damping_d);
            energy -= s6 * c6ij * damp / std::pow(r, 6);
        }
    }

    return energy;
}

DispersionResult compute_dispersion_energy(const std::vector<libint2::Atom>& atoms,
                                           const DispersionOptions& options) {
    DispersionResult result;
    result.method = options.method;
    result.method_label = to_string(options.method);
    result.damping_d = options.damping_d;

    switch (options.method) {
        case DispersionMethod::None:
            result.s6 = 0.0;
            result.energy = 0.0;
            return result;

        case DispersionMethod::D2: {
            const double s6 = std::isfinite(options.s6)
                ? options.s6
                : d2_default_s6_for_functional(options.functional);
            result.s6 = s6;
            result.energy = compute_d2_dispersion_energy(atoms, s6, options.damping_d);
            return result;
        }
    }

    throw std::runtime_error("unknown dispersion method");
}

}  // namespace miniqc::dispersion
