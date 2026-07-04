#include <Eigen/Dense>
#include <libint2/cxxapi.h>

#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "calculation_config.hpp"
#include "miniqc.hpp"
#include "geometry_optimizer.hpp"
#include "analytic_gradient.hpp"
#include "mp2_v3_direct_t1.hpp"
#include "fci.hpp"
#include "dft_grid.hpp"
#include "dft_lda.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using miniqc::CIType;
using miniqc::DriverConfig;
using miniqc::GradientMode;
using miniqc::Molecule;

void print_usage(const char* exe) {
    std::cerr
        << "Usage:\n"
        << "  " << exe << " input.in\n"
        << "  " << exe << " --config input.in\n"
        << "  " << exe << " --write-template-config > input.in\n\n"
        << "The configuration file uses simple INI-style key = value sections.\n";
}

miniqc::RHFOptions make_rhf_options(const DriverConfig& cfg, bool verbose_override) {
    miniqc::RHFOptions options;
    options.max_iter = cfg.scf_max_iter;
    options.e_conv = cfg.scf_e_conv;
    options.d_conv = cfg.scf_d_conv;
    options.use_diis = cfg.scf_use_diis;
    options.diis_start = cfg.scf_diis_start;
    options.diis_max_vec = cfg.scf_diis_max_vec;
    options.verbose = verbose_override;
    return options;
}

miniqc::GeometryOptions make_geometry_options(const DriverConfig& cfg) {
    miniqc::GeometryOptions options;
    options.max_steps = cfg.geom_max_steps;
    options.finite_difference_step = cfg.geom_finite_difference_step;
    options.max_step = cfg.geom_max_step;
    options.energy_conv = cfg.geom_energy_conv;
    options.max_force_conv = cfg.geom_max_force_conv;
    options.rms_force_conv = cfg.geom_rms_force_conv;
    options.verbose = cfg.geom_verbose;
    return options;
}

dft::LDAFunctional parse_dft_functional(const std::string& name) {
    const std::string v = miniqc::lowercase_copy(miniqc::trim_copy(name));
    if (v == "slater" || v == "slater_x" || v == "xalpha") {
        return dft::LDAFunctional::SlaterX;
    }
    if (v == "lda_x" || v == "libxc_lda_x") {
        return dft::LDAFunctional::Libxc_LDA_X;
    }
    if (v == "lda_x_pz81" || v == "lda_pz81" || v == "lda_pz") {
        return dft::LDAFunctional::Libxc_LDA_X_PZ81C;
    }
    if (v == "pbe" || v == "gga_pbe" || v == "libxc_gga_pbe") {
        return dft::LDAFunctional::Libxc_GGA_PBE;
    }
    throw std::runtime_error("unknown DFT functional: " + name);
}

int ci_func_code(CIType type) {
    switch (type) {
        case CIType::FCI: return 0;
        case CIType::CISD: return 1;
        case CIType::CID: return 2;
        case CIType::None: break;
    }
    return -1;
}

void validate_config(const DriverConfig& cfg) {
    if (cfg.multiplicity != 1) {
        throw std::runtime_error("this driver currently supports closed-shell multiplicity=1 only");
    }
    if (cfg.optimize_geometry && (cfg.run_mp2 || cfg.ci_type != CIType::None || cfg.run_dft)) {
        throw std::runtime_error(
            "geometry optimization is currently RHF-only; run optimization first, then run MP2/CI/DFT on the optimized XYZ");
    }
}

void print_job_summary(const DriverConfig& cfg, const Molecule& mol, int nelec) {
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "=== miniqc configuration ===\n";
    std::cout << "basis          = " << cfg.basis_name << "\n";
    std::cout << "xyz            = " << cfg.xyz_file << "\n";
    std::cout << "input unit     = " << cfg.coord_unit << "\n";
    std::cout << "charge         = " << mol.charge << "\n";
    std::cout << "multiplicity   = " << mol.multiplicity << "\n";
    std::cout << "nelec          = " << nelec << "\n";
    std::cout << "job            = " << (cfg.optimize_geometry ? "geometry optimization" : "single point") << "\n";
    if (cfg.optimize_geometry) {
        std::cout << "gradient       = " << miniqc::gradient_mode_name(cfg.gradient_mode) << "\n";
        std::cout << "opt output     = " << cfg.opt_output_xyz << "\n";
    } else {
        std::cout << "MP2            = " << (cfg.run_mp2 ? "yes" : "no") << "\n";
        std::cout << "CI             = " << miniqc::ci_type_name(cfg.ci_type) << "\n";
        std::cout << "DFT/RKS        = " << (cfg.run_dft ? "yes" : "no") << "\n";
    }
#ifdef _OPENMP
    std::cout << "OpenMP threads = " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP         = disabled\n";
#endif
    std::cout << "\n";
}

miniqc::RHFResult run_rhf_for_molecule(const Molecule& mol,
                                       const DriverConfig& cfg,
                                       bool verbose,
                                       const Eigen::MatrixXd* C_guess = nullptr,
                                       Eigen::MatrixXd* C_out = nullptr) {
    const int nelec = miniqc::electron_count(mol);
    miniqc::BasisContext ctx(mol, cfg.basis_name, true);
    const miniqc::OneBodyIntegrals one = miniqc::build_one_body_integrals(ctx.basis, mol.atoms);
    miniqc::RHFResult rhf = miniqc::rhf_closed_shell(
        ctx.basis,
        one.S,
        one.Hcore,
        nelec,
        mol,
        make_rhf_options(cfg, verbose),
        C_guess
    );

    if (C_out != nullptr) {
        *C_out = rhf.C;
    }
    return rhf;
}

double rhf_total_energy_for_geometry(const Molecule& mol,
                                     const DriverConfig& cfg,
                                     const Eigen::MatrixXd* C_guess,
                                     Eigen::MatrixXd* C_out) {
    return run_rhf_for_molecule(mol, cfg, false, C_guess, C_out).energy_total;
}

Eigen::VectorXd rhf_analytic_gradient_for_geometry(const Molecule& mol,
                                                   const DriverConfig& cfg,
                                                   const Eigen::MatrixXd* C_guess,
                                                   Eigen::MatrixXd* C_out) {
    const int nelec = miniqc::electron_count(mol);
    if (nelec % 2 != 0) throw std::runtime_error("closed-shell RHF gradient needs an even electron count");

    miniqc::BasisContext ctx(mol, cfg.basis_name, true);
    const miniqc::OneBodyIntegrals one = miniqc::build_one_body_integrals(ctx.basis, mol.atoms);
    miniqc::RHFResult rhf = miniqc::rhf_closed_shell(
        ctx.basis,
        one.S,
        one.Hcore,
        nelec,
        mol,
        make_rhf_options(cfg, false),
        C_guess
    );

    if (C_out != nullptr) {
        *C_out = rhf.C;
    }

    const Eigen::MatrixXd grad = miniqc::rhf_analytic_gradient(
        mol.atoms,
        ctx.basis,
        rhf.D,
        rhf.C,
        rhf.eps,
        nelec / 2
    );

    return miniqc::flatten_atom_gradient_rowwise(grad);
}

void run_geometry_optimization(const Molecule& mol0, const DriverConfig& cfg) {
    const Eigen::VectorXd x0 = miniqc::molecule_to_coordinate_vector(mol0);
    const miniqc::GeometryOptions geom_options = make_geometry_options(cfg);

    miniqc::EnergyFunction energy = [&](const Eigen::VectorXd& x,
                                        const Eigen::MatrixXd* C_guess,
                                        Eigen::MatrixXd* C_out) -> double {
        const Molecule trial = miniqc::molecule_with_coordinate_vector(mol0, x);
        return rhf_total_energy_for_geometry(trial, cfg, C_guess, C_out);
    };

    miniqc::GradientFunction gradient;
    if (cfg.gradient_mode == GradientMode::Analytic) {
        gradient = [&](const Eigen::VectorXd& x,
                       const Eigen::MatrixXd* C_guess,
                       Eigen::MatrixXd* C_out) -> Eigen::VectorXd {
            const Molecule trial = miniqc::molecule_with_coordinate_vector(mol0, x);
            return rhf_analytic_gradient_for_geometry(trial, cfg, C_guess, C_out);
        };
    }

    std::cout << "=== RHF geometry optimization ===\n";
    std::cout << "gradient mode = " << miniqc::gradient_mode_name(cfg.gradient_mode) << "\n\n";

    const miniqc::GeometryResult opt = miniqc::optimize_bfgs(x0, energy, geom_options, gradient);
    const Molecule opt_mol = miniqc::molecule_with_coordinate_vector(mol0, opt.x);

    std::cout << "\n=== Geometry optimization result ===\n";
    std::cout << "converged   = " << (opt.converged ? "yes" : "no") << "\n";
    std::cout << "steps       = " << opt.iterations << "\n";
    std::cout << "E_total     = " << std::setprecision(12) << opt.energy << " Ha\n";
    std::cout << "max|grad|   = " << opt.gradient.cwiseAbs().maxCoeff() << " Ha/bohr\n";
    std::cout << "rms_grad    = " << opt.gradient.norm() / std::sqrt(static_cast<double>(opt.gradient.size())) << " Ha/bohr\n";

    miniqc::write_xyz_angstrom(
        cfg.opt_output_xyz,
        opt_mol,
        "optimized by miniqc RHF BFGS; coordinates in Angstrom"
    );
    std::cout << "optimized geometry written to " << cfg.opt_output_xyz << "\n";

    if (!opt.converged) {
        throw std::runtime_error("geometry optimization did not converge");
    }
}

void print_rhf_result(const miniqc::RHFResult& rhf, const DriverConfig& cfg) {
    std::cout << "\n=== Final RHF result ===\n";
    std::cout << "converged   = " << (rhf.converged ? "yes" : "no") << "\n";
    std::cout << "iterations  = " << rhf.iterations << "\n";
    std::cout << std::setprecision(12);
    std::cout << "E_elec      = " << rhf.energy_electronic << " Ha\n";
    std::cout << "E_nuc       = " << rhf.energy_nuclear << " Ha\n";
    std::cout << "E_total     = " << rhf.energy_total << " Ha\n";

    if (cfg.print_orbitals) {
        std::cout << "\norbital eps:\n" << rhf.eps.transpose() << "\n";
    }
}

void maybe_print_gradient(const Molecule& mol,
                          const DriverConfig& cfg,
                          const miniqc::RHFResult& rhf,
                          const miniqc::BasisContext& ctx,
                          int nelec) {
    if (!cfg.print_gradient) return;

    const Eigen::MatrixXd grad = miniqc::rhf_analytic_gradient(
        mol.atoms,
        ctx.basis,
        rhf.D,
        rhf.C,
        rhf.eps,
        nelec / 2
    );

    std::cout << "\n=== RHF analytic gradient ===\n";
    std::cout << "dE/dR, Hartree/bohr:\n" << std::scientific << std::setprecision(12) << grad << "\n";
    std::cout << "force = -dE/dR, Hartree/bohr:\n" << -grad << "\n";
    std::cout << "gradient max abs = " << grad.cwiseAbs().maxCoeff() << std::fixed << "\n";
}

void run_mp2(const miniqc::BasisContext& ctx,
             const miniqc::RHFResult& rhf,
             const DriverConfig& cfg,
             int nocc) {
    if (!cfg.run_mp2) return;

    miniqc::MP2Options options;
    options.verbose = true;
    options.max_intermediate_mb = cfg.mp2_max_intermediate_mb;

    const miniqc::MP2Result mp2 = miniqc::compute_rhf_mp2_energy(
        ctx.basis, rhf.C, rhf.eps, nocc, rhf.energy_total, options);

    std::cout << "\n=== RHF-MP2 result ===\n";
    std::cout << "nbf         = " << mp2.nbf << "\n";
    std::cout << "nocc        = " << mp2.nocc << "\n";
    std::cout << "nvir        = " << mp2.nvir << "\n";
    std::cout << std::setprecision(12);
    std::cout << "E_HF        = " << rhf.energy_total << " Ha\n";
    std::cout << "E_MP2_corr  = " << mp2.correlation_energy << " Ha\n";
    std::cout << "E_MP2_total = " << mp2.total_energy << " Ha\n";
}

void run_ci(const miniqc::BasisContext& ctx,
            const miniqc::OneBodyIntegrals& one,
            const miniqc::RHFResult& rhf,
            const DriverConfig& cfg,
            int nelec) {
    if (cfg.ci_type == CIType::None) return;

    miniqc::FCIOptions options;
    options.verbose = true;
    options.max_determinants = cfg.ci_max_determinants;
    options.max_nmo = cfg.ci_max_nmo;
    options.max_tensor_mb = cfg.ci_max_tensor_mb;
    options.print_largest_coefficients = cfg.print_ci_coefficients;
    options.n_coefficients_to_print = cfg.ci_n_coefficients_to_print;

    const int func = ci_func_code(cfg.ci_type);
    const miniqc::FCIResult ci = miniqc::compute_fci_energy(
        ctx.basis,
        one.Hcore,
        rhf.C,
        nelec,
        rhf.energy_nuclear,
        options,
        func
    );

    std::cout << "\n=== " << miniqc::ci_type_name(cfg.ci_type) << " result ===\n";
    std::cout << "nmo         = " << ci.nmo << "\n";
    std::cout << "nelec       = " << ci.nelec << "\n";
    std::cout << "nalpha      = " << ci.nalpha << "\n";
    std::cout << "nbeta       = " << ci.nbeta << "\n";
    std::cout << "ndet        = " << ci.ndet << "\n";
    std::cout << std::setprecision(12);
    std::cout << "E_elec      = " << ci.electronic_energy << " Ha\n";
    std::cout << "E_nuc       = " << rhf.energy_nuclear << " Ha\n";
    std::cout << "E_total     = " << ci.total_energy << " Ha\n";
}

void run_dft(const Molecule& mol,
             const miniqc::BasisContext& ctx,
             const miniqc::OneBodyIntegrals& one,
             const miniqc::RHFResult& rhf,
             const DriverConfig& cfg,
             int nelec) {
    if (!cfg.run_dft) return;

    dft::AtomCenteredGridOptions grid_options;
    grid_options.n_radial = cfg.dft_n_radial;
    grid_options.angular_grid = cfg.dft_angular_grid;
    grid_options.r_max = cfg.dft_r_max;
    grid_options.radial_type = dft::RadialGridType::PowerMapped;
    grid_options.radial_power = cfg.dft_radial_power;
    grid_options.partition_type = dft::PartitionType::Becke;
    grid_options.becke_smooth_order = 3;
    grid_options.partition_eps = 1.0e-10;
    grid_options.partition_power = 4.0;
    grid_options.weight_cutoff = 0.0;
    grid_options.verbose = cfg.dft_verbose;

    const auto grid = dft::make_atom_centered_grid(mol.atoms, grid_options);

    dft::RKSLDAOptions options;
    options.max_iter = cfg.dft_max_iter;
    options.e_conv = cfg.dft_e_conv;
    options.d_conv = cfg.dft_d_conv;
    options.density_mixing = cfg.dft_density_mixing;
    options.verbose = cfg.dft_verbose;
    options.functional = parse_dft_functional(cfg.dft_functional);

    const auto rks = dft::run_rks_lda_exchange_only_sp(
        ctx.basis,
        one.S,
        one.Hcore,
        nelec,
        rhf.energy_nuclear,
        grid,
        options
    );

    std::cout << "\n=== RKS/DFT result ===\n";
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "functional = " << cfg.dft_functional << "\n";
    std::cout << "converged  = " << (rks.converged ? "yes" : "no") << "\n";
    std::cout << "niter      = " << rks.niter << "\n";
    std::cout << "E_x        = " << rks.E_x << " Ha\n";
    std::cout << "E_c        = " << rks.E_c << " Ha\n";
    std::cout << "E_xc       = " << rks.E_xc << " Ha\n";
    std::cout << "Ne(grid)   = " << rks.Ne_grid << "\n";
    std::cout << "E_total    = " << rks.E_total << " Ha\n";
}

void run_single_point(const Molecule& mol, const DriverConfig& cfg) {
    const int nelec = miniqc::electron_count(mol);
    if (nelec <= 0) throw std::runtime_error("non-positive electron count");
    if (nelec % 2 != 0) throw std::runtime_error("closed-shell RHF needs an even electron count");

    miniqc::BasisContext ctx(mol, cfg.basis_name, true);
    std::cout << "nshell = " << ctx.basis.size()
              << ", nbf = " << ctx.basis.nbf()
              << ", max_nprim = " << ctx.basis.max_nprim()
              << ", max_l = " << ctx.basis.max_l() << "\n";

    const miniqc::OneBodyIntegrals one = miniqc::build_one_body_integrals(ctx.basis, mol.atoms);

    if (cfg.print_matrices) {
        std::cout << "\nAtoms in bohr:\n";
        miniqc::print_atoms_bohr(mol.atoms);
        std::cout << "S:\n" << one.S << "\n\n";
        std::cout << "T:\n" << one.T << "\n\n";
        std::cout << "V:\n" << one.V << "\n\n";
        std::cout << "Hcore:\n" << one.Hcore << "\n\n";
    }

    const miniqc::RHFResult rhf = miniqc::rhf_closed_shell(
        ctx.basis,
        one.S,
        one.Hcore,
        nelec,
        mol,
        make_rhf_options(cfg, cfg.scf_verbose),
        nullptr
    );

    print_rhf_result(rhf, cfg);
    maybe_print_gradient(mol, cfg, rhf, ctx, nelec);

    if (cfg.print_matrices) {
        std::cout << "\ndensity D:\n" << rhf.D << "\n\n";
        std::cout << "MO C:\n" << rhf.C << "\n";
    }

    const int nocc = nelec / 2;
    run_mp2(ctx, rhf, cfg, nocc);
    run_ci(ctx, one, rhf, cfg, nelec);
    run_dft(mol, ctx, one, rhf, cfg, nelec);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--write-template-config") {
        miniqc::write_template_config(std::cout);
        return 0;
    }

    std::string config_file;
    if (argc == 2) {
        config_file = argv[1];
    } else if (argc == 3 && std::string(argv[1]) == "--config") {
        config_file = argv[2];
    } else {
        print_usage(argv[0]);
        return 1;
    }

    libint2::initialize();

    try {
        const DriverConfig cfg = miniqc::load_driver_config(config_file);
        validate_config(cfg);

        Molecule mol = miniqc::read_xyz_molecule(
            cfg.xyz_file,
            cfg.coord_unit,
            cfg.charge,
            cfg.multiplicity
        );

        const int nelec = miniqc::electron_count(mol);
        if (nelec <= 0) throw std::runtime_error("non-positive electron count");
        if (nelec % 2 != 0) throw std::runtime_error("closed-shell RHF needs an even electron count");

        print_job_summary(cfg, mol, nelec);

        if (cfg.optimize_geometry) {
            run_geometry_optimization(mol, cfg);
        } else {
            run_single_point(mol, cfg);
        }

        libint2::finalize();
        return 0;
    } catch (const std::exception& e) {
        libint2::finalize();
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
