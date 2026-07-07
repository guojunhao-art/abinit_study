#pragma once

#include <libint2/cxxapi.h>

#include <limits>
#include <string>
#include <vector>

namespace miniqc::dispersion {

enum class DispersionMethod {
    None,
    D2
};

struct D2AtomParameters {
    double c6_j_nm6_mol = 0.0;
    double r0_angstrom = 0.0;
};

struct DispersionOptions {
    DispersionMethod method = DispersionMethod::None;
    std::string functional;

    // NaN means use the functional-specific D2 default scale if available.
    double s6 = std::numeric_limits<double>::quiet_NaN();
    double damping_d = 20.0;
};

struct DispersionResult {
    DispersionMethod method = DispersionMethod::None;
    std::string method_label = "none";
    double s6 = 0.0;
    double damping_d = 20.0;
    double energy = 0.0;  // Hartree
};

std::string normalize_dispersion_name(const std::string& name);
DispersionMethod parse_dispersion_method(const std::string& name);
std::string to_string(DispersionMethod method);

D2AtomParameters d2_parameters_for_atomic_number(int atomic_number);
bool has_d2_parameters(int atomic_number);
double d2_c6_conversion_to_atomic_units();
double d2_default_s6_for_functional(const std::string& functional);

double compute_d2_dispersion_energy(const std::vector<libint2::Atom>& atoms,
                                    double s6,
                                    double damping_d = 20.0);

DispersionResult compute_dispersion_energy(const std::vector<libint2::Atom>& atoms,
                                           const DispersionOptions& options);

}  // namespace miniqc::dispersion
