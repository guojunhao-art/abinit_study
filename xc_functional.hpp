#pragma once

#include <string>
#include <vector>

namespace miniqc::xc {

enum class XCFamily {
    LDA,
    GGA,
    MGGA,
    HybridGGA,
    HybridMGGA,
    RangeSeparatedHybridGGA,
    RangeSeparatedHybridMGGA
};

enum class SpinMode {
    Restricted,
    Unrestricted
};

struct XCRequirements {
    bool needs_rho = true;
    bool needs_sigma = false;      // GGA ingredient: |grad rho|^2 for RKS
    bool needs_laplacian = false;  // reserved for future meta-GGA support
    bool needs_tau = false;        // meta-GGA ingredient: kinetic energy density
};

struct XCComponent {
    int libxc_id = 0;
    double scale = 1.0;
    std::string label;
};

struct XCFunctional {
    std::string name;
    XCFamily family = XCFamily::LDA;
    SpinMode spin_mode = SpinMode::Restricted;
    XCRequirements requirements;

    // Fraction of full-range exact exchange used by the generalized KS Fock.
    // For pure LDA/GGA/meta-GGA functionals this is zero.
    double exact_exchange_fraction = 0.0;

    // Reserved fields for range-separated hybrids.
    bool is_range_separated = false;
    double omega = 0.0;
    double alpha = 0.0;
    double beta = 0.0;

    std::vector<XCComponent> components;

    bool is_hybrid() const;
    bool is_gga_like() const;
    bool is_meta_gga() const;
};

std::string to_string(XCFamily family);
std::string to_string(SpinMode spin_mode);

// Construct a named functional descriptor.  This function only describes what
// the SCF/grid code must provide.  It does not evaluate the functional.
// Supported names in this first descriptor layer:
//   slater_x, lda_x, lda_x_pz81, pbe, blyp, b3lyp, pbe0, m06-2x
XCFunctional make_xc_functional(const std::string& name,
                                SpinMode spin_mode = SpinMode::Restricted);

// Convenience predicates for driver validation.
bool is_supported_by_current_rks_driver(const XCFunctional& functional);
std::string current_rks_support_message(const XCFunctional& functional);

}  // namespace miniqc::xc
