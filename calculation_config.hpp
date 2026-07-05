#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace miniqc {

enum class GradientMode {
    Analytic,
    Numerical
};

enum class CIType {
    None,
    FCI,
    CID,
    CISD
};

struct DriverConfig {
    // Molecule / basis.
    std::string basis_name;
    std::string xyz_file;
    std::string coord_unit = "angstrom";
    int charge = 0;
    int multiplicity = 1;

    // Single-point post-HF / DFT switches.
    bool run_mp2 = false;
    CIType ci_type = CIType::None;
    bool run_dft = false;

    // Output switches.
    bool print_matrices = false;
    bool print_orbitals = true;
    bool print_gradient = false;
    bool print_ci_coefficients = false;

    // RHF SCF options.
    int scf_max_iter = 128;
    double scf_e_conv = 1.0e-10;
    double scf_d_conv = 1.0e-8;
    bool scf_use_diis = true;
    int scf_diis_start = 2;
    std::size_t scf_diis_max_vec = 8;
    bool scf_verbose = true;

    // Geometry optimization.
    bool optimize_geometry = false;
    GradientMode gradient_mode = GradientMode::Analytic;
    std::string opt_output_xyz = "optimized.xyz";
    int geom_max_steps = 50;
    double geom_finite_difference_step = 1.0e-3;
    double geom_max_step = 0.20;
    double geom_energy_conv = 1.0e-7;
    double geom_max_force_conv = 4.5e-4;
    double geom_rms_force_conv = 3.0e-4;
    bool geom_verbose = true;

    // MP2.
    double mp2_max_intermediate_mb = 8192.0;

    // CI.
    std::size_t ci_max_determinants = 5000;
    std::size_t ci_max_nmo = 16;
    double ci_max_tensor_mb = 4096.0;
    std::size_t ci_n_coefficients_to_print = 12;

    // DFT/RKS grid path.  The driver now uses the total-Exc RKS implementation
    // in rks_xc.cpp.  Hybrid GGA functionals are routed through the same path.
    std::size_t dft_n_radial = 80;
    int dft_angular_grid = 26;
    double dft_r_max = 12.0;
    double dft_radial_power = 2.0;
    double dft_density_mixing = 0.3;
    int dft_max_iter = 256;
    double dft_e_conv = 1.0e-10;
    double dft_d_conv = 1.0e-8;
    bool dft_verbose = true;
    std::string dft_functional = "pbe";
};

inline std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

inline std::string lowercase_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline std::string strip_inline_comment(std::string s) {
    const std::size_t hash_pos = s.find('#');
    const std::size_t semi_pos = s.find(';');
    std::size_t pos = std::string::npos;
    if (hash_pos != std::string::npos) pos = hash_pos;
    if (semi_pos != std::string::npos) {
        pos = (pos == std::string::npos) ? semi_pos : std::min(pos, semi_pos);
    }
    if (pos != std::string::npos) s.erase(pos);
    return trim_copy(std::move(s));
}

class KeyValueConfig {
public:
    static KeyValueConfig read_file(const std::string& filename);

    bool has(std::initializer_list<const char*> keys) const {
        for (const char* key : keys) {
            if (values_.count(normalize_key(key)) != 0) return true;
        }
        return false;
    }

    std::string get_string(std::initializer_list<const char*> keys,
                           const std::string& fallback) const {
        for (const char* key : keys) {
            auto it = values_.find(normalize_key(key));
            if (it != values_.end()) return it->second;
        }
        return fallback;
    }

    std::string get_required_string(std::initializer_list<const char*> keys,
                                    const std::string& label) const {
        for (const char* key : keys) {
            auto it = values_.find(normalize_key(key));
            if (it != values_.end()) return it->second;
        }
        throw std::runtime_error("missing required config entry: " + label);
    }

    int get_int(std::initializer_list<const char*> keys, int fallback) const {
        const std::string s = get_string(keys, "");
        return s.empty() ? fallback : std::stoi(s);
    }

    std::size_t get_size(std::initializer_list<const char*> keys,
                         std::size_t fallback) const {
        const std::string s = get_string(keys, "");
        return s.empty() ? fallback : static_cast<std::size_t>(std::stoull(s));
    }

    double get_double(std::initializer_list<const char*> keys, double fallback) const {
        const std::string s = get_string(keys, "");
        return s.empty() ? fallback : std::stod(s);
    }

    bool get_bool(std::initializer_list<const char*> keys, bool fallback) const {
        const std::string s = get_string(keys, "");
        if (s.empty()) return fallback;
        const std::string v = lowercase_copy(trim_copy(s));
        if (v == "1" || v == "true" || v == "yes" || v == "y" || v == "on") return true;
        if (v == "0" || v == "false" || v == "no" || v == "n" || v == "off") return false;
        throw std::runtime_error("invalid boolean config value: " + s);
    }

    void set(std::string key, std::string value) {
        values_[normalize_key(std::move(key))] = trim_copy(std::move(value));
    }

private:
    static std::string normalize_key(std::string key) {
        key = trim_copy(std::move(key));
        key = lowercase_copy(std::move(key));
        key.erase(std::remove_if(key.begin(), key.end(), [](unsigned char c) {
            return std::isspace(c);
        }), key.end());
        return key;
    }

    std::unordered_map<std::string, std::string> values_;
};

inline KeyValueConfig KeyValueConfig::read_file(const std::string& filename) {
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("cannot open config file: " + filename);

    KeyValueConfig cfg;
    std::string section;
    std::string line;
    int line_no = 0;

    while (std::getline(input, line)) {
        ++line_no;
        line = strip_inline_comment(line);
        if (line.empty()) continue;

        if (line.front() == '[' && line.back() == ']') {
            section = lowercase_copy(trim_copy(line.substr(1, line.size() - 2)));
            continue;
        }

        std::size_t pos = line.find('=');
        if (pos == std::string::npos) pos = line.find(':');
        if (pos == std::string::npos) {
            std::ostringstream oss;
            oss << "invalid config line " << line_no << ": " << line;
            throw std::runtime_error(oss.str());
        }

        std::string key = trim_copy(line.substr(0, pos));
        std::string value = trim_copy(line.substr(pos + 1));
        if (key.empty()) {
            std::ostringstream oss;
            oss << "empty config key at line " << line_no;
            throw std::runtime_error(oss.str());
        }

        if (!section.empty() && key.find('.') == std::string::npos) {
            key = section + "." + key;
        }
        cfg.set(std::move(key), std::move(value));
    }

    return cfg;
}

inline GradientMode parse_gradient_mode(const std::string& value) {
    const std::string v = lowercase_copy(trim_copy(value));
    if (v == "analytic" || v == "analytical" || v == "rhf_analytic") return GradientMode::Analytic;
    if (v == "numerical" || v == "numeric" || v == "finite_difference" || v == "fd") return GradientMode::Numerical;
    throw std::runtime_error("unknown geometry gradient mode: " + value);
}

inline CIType parse_ci_type(const std::string& value) {
    const std::string v = lowercase_copy(trim_copy(value));
    if (v.empty() || v == "none" || v == "off" || v == "false" || v == "0") return CIType::None;
    if (v == "fci") return CIType::FCI;
    if (v == "cid") return CIType::CID;
    if (v == "cisd") return CIType::CISD;
    throw std::runtime_error("unknown CI type: " + value);
}

inline std::string ci_type_name(CIType type) {
    switch (type) {
        case CIType::None: return "none";
        case CIType::FCI: return "FCI";
        case CIType::CID: return "CID";
        case CIType::CISD: return "CISD";
    }
    return "unknown";
}

inline std::string gradient_mode_name(GradientMode mode) {
    switch (mode) {
        case GradientMode::Analytic: return "analytic";
        case GradientMode::Numerical: return "numerical";
    }
    return "unknown";
}

inline DriverConfig load_driver_config(const std::string& filename) {
    const KeyValueConfig kv = KeyValueConfig::read_file(filename);
    DriverConfig cfg;

    cfg.basis_name = kv.get_required_string({"molecule.basis", "basis"}, "molecule.basis");
    cfg.xyz_file = kv.get_required_string({"molecule.xyz", "molecule.geometry", "xyz", "geometry"}, "molecule.xyz");
    cfg.coord_unit = kv.get_string({"molecule.unit", "unit"}, cfg.coord_unit);
    cfg.charge = kv.get_int({"molecule.charge", "charge"}, cfg.charge);
    cfg.multiplicity = kv.get_int({"molecule.multiplicity", "multiplicity"}, cfg.multiplicity);

    const std::string job = lowercase_copy(kv.get_string({"calculation.job", "job"}, "single_point"));
    cfg.optimize_geometry = (job == "opt" || job == "optimize" || job == "optimization" || job == "geomopt");
    cfg.optimize_geometry = kv.get_bool({"geometry.optimize", "optimize", "opt"}, cfg.optimize_geometry);

    cfg.run_mp2 = kv.get_bool({"calculation.mp2", "mp2"}, cfg.run_mp2);
    cfg.ci_type = parse_ci_type(kv.get_string({"calculation.ci", "ci"}, ci_type_name(cfg.ci_type)));
    cfg.run_dft = kv.get_bool({"calculation.dft", "dft", "rks"}, cfg.run_dft);

    cfg.print_matrices = kv.get_bool({"output.print_matrices", "print_matrices"}, cfg.print_matrices);
    cfg.print_orbitals = kv.get_bool({"output.print_orbitals", "print_orbitals"}, cfg.print_orbitals);
    cfg.print_gradient = kv.get_bool({"output.print_gradient", "print_gradient"}, cfg.print_gradient);
    cfg.print_ci_coefficients = kv.get_bool({"ci.print_coefficients", "output.print_ci_coefficients", "print_ci_coefficients"}, cfg.print_ci_coefficients);

    cfg.scf_max_iter = kv.get_int({"scf.max_iter"}, cfg.scf_max_iter);
    cfg.scf_e_conv = kv.get_double({"scf.e_conv"}, cfg.scf_e_conv);
    cfg.scf_d_conv = kv.get_double({"scf.d_conv"}, cfg.scf_d_conv);
    cfg.scf_use_diis = kv.get_bool({"scf.use_diis"}, cfg.scf_use_diis);
    cfg.scf_diis_start = kv.get_int({"scf.diis_start"}, cfg.scf_diis_start);
    cfg.scf_diis_max_vec = kv.get_size({"scf.diis_max_vec", "scf.diis_max_vecs"}, cfg.scf_diis_max_vec);
    cfg.scf_verbose = kv.get_bool({"scf.verbose"}, cfg.scf_verbose);

    cfg.gradient_mode = parse_gradient_mode(kv.get_string({"geometry.gradient", "gradient"}, gradient_mode_name(cfg.gradient_mode)));
    cfg.opt_output_xyz = kv.get_string({"geometry.output", "geometry.opt_output", "opt_output"}, cfg.opt_output_xyz);
    cfg.geom_max_steps = kv.get_int({"geometry.max_steps"}, cfg.geom_max_steps);
    cfg.geom_finite_difference_step = kv.get_double({"geometry.finite_difference_step", "geometry.fd_step"}, cfg.geom_finite_difference_step);
    cfg.geom_max_step = kv.get_double({"geometry.max_step"}, cfg.geom_max_step);
    cfg.geom_energy_conv = kv.get_double({"geometry.energy_conv"}, cfg.geom_energy_conv);
    cfg.geom_max_force_conv = kv.get_double({"geometry.max_force_conv"}, cfg.geom_max_force_conv);
    cfg.geom_rms_force_conv = kv.get_double({"geometry.rms_force_conv"}, cfg.geom_rms_force_conv);
    cfg.geom_verbose = kv.get_bool({"geometry.verbose"}, cfg.geom_verbose);

    cfg.mp2_max_intermediate_mb = kv.get_double({"mp2.max_intermediate_mb"}, cfg.mp2_max_intermediate_mb);

    cfg.ci_max_determinants = kv.get_size({"ci.max_determinants"}, cfg.ci_max_determinants);
    cfg.ci_max_nmo = kv.get_size({"ci.max_nmo"}, cfg.ci_max_nmo);
    cfg.ci_max_tensor_mb = kv.get_double({"ci.max_tensor_mb"}, cfg.ci_max_tensor_mb);
    cfg.ci_n_coefficients_to_print = kv.get_size({"ci.n_coefficients_to_print"}, cfg.ci_n_coefficients_to_print);

    cfg.dft_n_radial = kv.get_size({"dft.n_radial"}, cfg.dft_n_radial);
    cfg.dft_angular_grid = kv.get_int({"dft.angular_grid"}, cfg.dft_angular_grid);
    cfg.dft_r_max = kv.get_double({"dft.r_max"}, cfg.dft_r_max);
    cfg.dft_radial_power = kv.get_double({"dft.radial_power"}, cfg.dft_radial_power);
    cfg.dft_density_mixing = kv.get_double({"dft.density_mixing"}, cfg.dft_density_mixing);
    cfg.dft_max_iter = kv.get_int({"dft.max_iter"}, cfg.dft_max_iter);
    cfg.dft_e_conv = kv.get_double({"dft.e_conv"}, cfg.dft_e_conv);
    cfg.dft_d_conv = kv.get_double({"dft.d_conv"}, cfg.dft_d_conv);
    cfg.dft_verbose = kv.get_bool({"dft.verbose"}, cfg.dft_verbose);
    cfg.dft_functional = kv.get_string({"dft.functional"}, cfg.dft_functional);

    if (cfg.basis_name.empty()) throw std::runtime_error("basis name cannot be empty");
    if (cfg.xyz_file.empty()) throw std::runtime_error("XYZ filename cannot be empty");
    return cfg;
}

inline void write_template_config(std::ostream& out) {
    out << R"ini(# miniqc configuration file

[molecule]
basis = 6-311g**
xyz = water.xyz
unit = angstrom
charge = 0
multiplicity = 1

[calculation]
# job = single_point or optimize
job = single_point
mp2 = false
# ci = none, fci, cid, or cisd
ci = none
dft = false

[output]
print_matrices = false
print_orbitals = true
print_gradient = false

[scf]
max_iter = 128
e_conv = 1.0e-10
d_conv = 1.0e-8
use_diis = true
diis_start = 2
diis_max_vec = 8
verbose = true

[geometry]
# Used only when calculation.job = optimize or geometry.optimize = true.
# gradient = analytic or numerical
gradient = analytic
output = optimized.xyz
max_steps = 50
finite_difference_step = 1.0e-3
max_step = 0.20
energy_conv = 1.0e-7
max_force_conv = 4.5e-4
rms_force_conv = 3.0e-4
verbose = true

[mp2]
max_intermediate_mb = 8192.0

[ci]
max_determinants = 5000
max_nmo = 16
max_tensor_mb = 4096.0
print_coefficients = false
n_coefficients_to_print = 12

[dft]
# Total-XC RKS path. Examples: slater_x, lda_x, lda_x_pz81, pbe, blyp, b3lyp, pbe0.
# M06-2X is recognized by XCFunctional but still requires meta-GGA tau support.
functional = pbe
n_radial = 80
angular_grid = 26
r_max = 12.0
radial_power = 2.0
density_mixing = 0.3
max_iter = 256
e_conv = 1.0e-10
d_conv = 1.0e-8
verbose = true
)ini";
}

}  // namespace miniqc
