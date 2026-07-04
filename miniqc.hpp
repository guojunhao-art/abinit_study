#pragma once

// Convenience aggregate header for the staged refactor.
//
// The historical executable still includes several module headers directly.
// New small drivers/tests can include this file to access the shared core,
// integral, SCF, RHF, and XC helper layers introduced during the refactor.

#include "basis_context.hpp"
#include "core_molecule.hpp"
#include "dft_xc_matrix.hpp"
#include "linalg_utils.hpp"
#include "one_body_integrals.hpp"
#include "periodic_table.hpp"
#include "rhf.hpp"
#include "scf_diis.hpp"
#include "two_body_fock.hpp"
#include "xc_evaluator.hpp"
#include "xc_functional.hpp"
#include "xyz_io.hpp"
