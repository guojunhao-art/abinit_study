#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

static inline std::uint64_t bit64(int p) {
    return std::uint64_t{1} << p;
}

static std::vector<int> range_int(int begin, int end) {
    std::vector<int> v;
    for (int i = begin; i < end; ++i) v.push_back(i);
    return v;
}

static std::uint64_t lowest_occupied_bits(int nocc) {
    if (nocc < 0 || nocc > 63) {
        throw std::runtime_error("invalid occupation count");
    }
    if (nocc == 0) return 0ULL;
    return (std::uint64_t{1} << nocc) - 1ULL;
}

std::vector<std::uint64_t> generate_cisd_determinants(
    int nmo,
    int nalpha,
    int nbeta,
    bool include_singles
) {
    if (nmo <= 0 || nmo > 32) {
        throw std::runtime_error("current uint64 determinant layout requires nmo <= 32");
    }
    if (nalpha < 0 || nalpha > nmo || nbeta < 0 || nbeta > nmo) {
        throw std::runtime_error("invalid nalpha/nbeta for CISD determinant generation");
    }

    const std::uint64_t alpha_ref = lowest_occupied_bits(nalpha);
    const std::uint64_t beta_ref  = lowest_occupied_bits(nbeta);

    auto pack_det = [&](std::uint64_t alpha_bits, std::uint64_t beta_bits) {
        return alpha_bits | (beta_bits << nmo);
    };

    const std::uint64_t ref_det = pack_det(alpha_ref, beta_ref);

    std::vector<std::uint64_t> dets;
    std::unordered_set<std::uint64_t> seen;

    auto add_det = [&](std::uint64_t det) {
        if (seen.insert(det).second) {
            dets.push_back(det);
        }
    };

    add_det(ref_det);

    const std::vector<int> occ_a  = range_int(0, nalpha);
    const std::vector<int> virt_a = range_int(nalpha, nmo);

    const std::vector<int> occ_b  = range_int(0, nbeta);
    const std::vector<int> virt_b = range_int(nbeta, nmo);

    // -------------------------
    // Singles
    // -------------------------
    if (include_singles) {
        // alpha singles: i_alpha -> a_alpha
        for (int i : occ_a) {
            for (int a : virt_a) {
                const std::uint64_t alpha_new =
                    alpha_ref ^ bit64(i) ^ bit64(a);

                add_det(pack_det(alpha_new, beta_ref));
            }
        }

        // beta singles: i_beta -> a_beta
        for (int i : occ_b) {
            for (int a : virt_b) {
                const std::uint64_t beta_new =
                    beta_ref ^ bit64(i) ^ bit64(a);

                add_det(pack_det(alpha_ref, beta_new));
            }
        }
    }

    // -------------------------
    // Alpha-alpha doubles
    // i,j alpha -> a,b alpha
    // Use i<j and a<b to avoid duplicates.
    // -------------------------
    for (std::size_t ii = 0; ii < occ_a.size(); ++ii) {
        for (std::size_t jj = ii + 1; jj < occ_a.size(); ++jj) {
            const int i = occ_a[ii];
            const int j = occ_a[jj];

            for (std::size_t aa = 0; aa < virt_a.size(); ++aa) {
                for (std::size_t bb = aa + 1; bb < virt_a.size(); ++bb) {
                    const int a = virt_a[aa];
                    const int b = virt_a[bb];

                    const std::uint64_t alpha_new =
                        alpha_ref ^ bit64(i) ^ bit64(j) ^ bit64(a) ^ bit64(b);

                    add_det(pack_det(alpha_new, beta_ref));
                }
            }
        }
    }

    // -------------------------
    // Beta-beta doubles
    // i,j beta -> a,b beta
    // -------------------------
    for (std::size_t ii = 0; ii < occ_b.size(); ++ii) {
        for (std::size_t jj = ii + 1; jj < occ_b.size(); ++jj) {
            const int i = occ_b[ii];
            const int j = occ_b[jj];

            for (std::size_t aa = 0; aa < virt_b.size(); ++aa) {
                for (std::size_t bb = aa + 1; bb < virt_b.size(); ++bb) {
                    const int a = virt_b[aa];
                    const int b = virt_b[bb];

                    const std::uint64_t beta_new =
                        beta_ref ^ bit64(i) ^ bit64(j) ^ bit64(a) ^ bit64(b);

                    add_det(pack_det(alpha_ref, beta_new));
                }
            }
        }
    }

    // -------------------------
    // Alpha-beta doubles
    // i alpha -> a alpha, j beta -> b beta
    // -------------------------
    for (int i : occ_a) {
        for (int a : virt_a) {
            const std::uint64_t alpha_new =
                alpha_ref ^ bit64(i) ^ bit64(a);

            for (int j : occ_b) {
                for (int b : virt_b) {
                    const std::uint64_t beta_new =
                        beta_ref ^ bit64(j) ^ bit64(b);

                    add_det(pack_det(alpha_new, beta_new));
                }
            }
        }
    }

    return dets;
}
