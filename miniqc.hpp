#pragma once

// Convenience aggregate header for the staged refactor.
//
// The historical executable still includes several module headers directly.
// New small drivers/tests can include this file to access the shared core,
// integral, SCF, and RHF helpers introduced during the refactor.

#include "basis_context.hpp"
#include "core_molecule.hpp"
#include "linalg_utils.hpp"
#include "one_body_integrals.hpp"
#include "periodic_table.hpp"
#include "rhf.hpp"
#include "scf_diis.hpp"
#include "two_body_fock.hpp"
#include "xyz_io.hpp"
