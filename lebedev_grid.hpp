#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace dft {
namespace lebedev {

struct LebedevPoint {
    Eigen::Vector3d u;
    double w = 0.0;  // angular weight, normally sum to 4*pi
};

enum class OrbitType {
    A1,   // (1,0,0), 6 points
    A2,   // (0,a,a), a = 1/sqrt(2), 12 points
    A3,   // (a,a,a), a = 1/sqrt(3), 8 points
    AAB,  // (a,a,b), b = sqrt(1 - 2a^2), 24 points
    AB0,  // (a,b,0), b = sqrt(1 - a^2), 24 points
    ABC   // (a,b,c), c = sqrt(1 - a^2 - b^2), 48 points
};

struct RuleTerm {
    OrbitType type;
    double a;
    double b;
    double w;  // normalized weight, sum over all expanded points = 1
};

inline double pi() {
    return 3.141592653589793238462643383279502884;
}

inline bool nearly_equal(double x, double y, double tol = 1.0e-14) {
    return std::abs(x - y) < tol;
}

inline void push_unique_point(std::vector<LebedevPoint>& points,
                              const Eigen::Vector3d& u,
                              double w) {
    for (const auto& p : points) {
        if ((p.u - u).norm() < 1.0e-13) {
            return;
        }
    }

    LebedevPoint p;
    p.u = u;
    p.w = w;
    points.push_back(p);
}

inline std::vector<std::array<double, 3>>
unique_permutations(double x, double y, double z) {
    std::array<double, 3> base = {x, y, z};
    std::sort(base.begin(), base.end());

    std::vector<std::array<double, 3>> perms;

    do {
        bool duplicate = false;
        for (const auto& p : perms) {
            if (nearly_equal(p[0], base[0]) &&
                nearly_equal(p[1], base[1]) &&
                nearly_equal(p[2], base[2])) {
                duplicate = true;
                break;
            }
        }

        if (!duplicate) {
            perms.push_back(base);
        }
    } while (std::next_permutation(base.begin(), base.end()));

    return perms;
}

inline void add_orbit(std::vector<LebedevPoint>& points,
                      double x,
                      double y,
                      double z,
                      double w) {
    const auto perms = unique_permutations(x, y, z);

    for (const auto& p : perms) {
        for (int sx : {-1, 1}) {
            for (int sy : {-1, 1}) {
                for (int sz : {-1, 1}) {
                    const double xx = nearly_equal(p[0], 0.0) ? 0.0 : sx * p[0];
                    const double yy = nearly_equal(p[1], 0.0) ? 0.0 : sy * p[1];
                    const double zz = nearly_equal(p[2], 0.0) ? 0.0 : sz * p[2];

                    Eigen::Vector3d u(xx, yy, zz);

                    const double norm = u.norm();
                    if (norm < 1.0e-14) {
                        throw std::runtime_error("Lebedev orbit generated zero vector");
                    }

                    u /= norm;

                    push_unique_point(points, u, w);
                }
            }
        }
    }
}

inline void add_rule_term(std::vector<LebedevPoint>& points,
                          const RuleTerm& term) {
    if (!std::isfinite(term.w)) {
        throw std::runtime_error("Lebedev rule contains non-finite weight");
    }

    switch (term.type) {
        case OrbitType::A1: {
            add_orbit(points, 1.0, 0.0, 0.0, term.w);
            break;
        }

        case OrbitType::A2: {
            const double a = 1.0 / std::sqrt(2.0);
            add_orbit(points, 0.0, a, a, term.w);
            break;
        }

        case OrbitType::A3: {
            const double a = 1.0 / std::sqrt(3.0);
            add_orbit(points, a, a, a, term.w);
            break;
        }

        case OrbitType::AAB: {
            const double a = term.a;
            const double b2 = 1.0 - 2.0 * a * a;
            if (b2 <= 0.0) {
                throw std::runtime_error("invalid Lebedev AAB orbit parameter");
            }
            const double b = std::sqrt(b2);
            add_orbit(points, a, a, b, term.w);
            break;
        }

        case OrbitType::AB0: {
            const double a = term.a;
            const double b2 = 1.0 - a * a;
            if (b2 <= 0.0) {
                throw std::runtime_error("invalid Lebedev AB0 orbit parameter");
            }
            const double b = std::sqrt(b2);
            add_orbit(points, a, b, 0.0, term.w);
            break;
        }

        case OrbitType::ABC: {
            const double a = term.a;
            const double b = term.b;
            const double c2 = 1.0 - a * a - b * b;
            if (c2 <= 0.0) {
                throw std::runtime_error("invalid Lebedev ABC orbit parameter");
            }
            const double c = std::sqrt(c2);
            add_orbit(points, a, b, c, term.w);
            break;
        }

        default:
            throw std::runtime_error("unknown Lebedev orbit type");
    }
}

inline std::vector<RuleTerm> rule_terms(int n_points) {
    switch (n_points) {
        case 6:
            // Degree 3.
            return {
                {OrbitType::A1, 0.0, 0.0, 0.1666666666666667}
            };

        case 14:
            // Degree 5.
            return {
                {OrbitType::A1, 0.0, 0.0, 0.06666666666666667},
                {OrbitType::A3, 0.0, 0.0, 0.07500000000000000}
            };

        case 26:
            // Degree 7.
            return {
                {OrbitType::A1, 0.0, 0.0, 0.04761904761904762},
                {OrbitType::A2, 0.0, 0.0, 0.03809523809523810},
                {OrbitType::A3, 0.0, 0.0, 0.03214285714285714}
            };

        case 38:
            // Degree 9.
            return {
                {OrbitType::A1, 0.0, 0.0, 0.009523809523809524},
                {OrbitType::A3, 0.0, 0.0, 0.03214285714285714},
                {OrbitType::AB0, 0.4597008433809831, 0.0, 0.02857142857142857}
            };

        case 50:
            // Degree 11.
            return {
                {OrbitType::A1, 0.0, 0.0, 0.01269841269841270},
                {OrbitType::A2, 0.0, 0.0, 0.02257495590828924},
                {OrbitType::A3, 0.0, 0.0, 0.02109375000000000},
                {OrbitType::AAB, 0.3015113445777636, 0.0, 0.02017333553791887}
            };

        case 74:
            // Degree 13.
            // Contains a negative A3 weight.
            return {
                {OrbitType::A1, 0.0, 0.0, 0.0005130671797338464},
                {OrbitType::A2, 0.0, 0.0, 0.01660406956574204},
                {OrbitType::A3, 0.0, 0.0, -0.02958603896103896},
                {OrbitType::AAB, 0.4803844614152614, 0.0, 0.02657620708215946},
                {OrbitType::AB0, 0.3207726489807764, 0.0, 0.01652217099371571}
            };

        case 86:
            // Degree 15.
            return {
                {OrbitType::A1, 0.0, 0.0, 0.01154401154401154},
                {OrbitType::A3, 0.0, 0.0, 0.01194390908585628},
                {OrbitType::AAB, 0.3696028464541502, 0.0, 0.01111055571060340},
                {OrbitType::AAB, 0.6943540066026664, 0.0, 0.01187650129453714},
                {OrbitType::AB0, 0.3742430390903412, 0.0, 0.01181230374690448}
            };

        case 110:
            // Degree 17.
            return {
                {OrbitType::A1, 0.0, 0.0, 0.003828270494937162},
                {OrbitType::A3, 0.0, 0.0, 0.009793737512487512},
                {OrbitType::AAB, 0.1851156353447362, 0.0, 0.008211737283191111},
                {OrbitType::AAB, 0.6904210483822922, 0.0, 0.009942814891178103},
                {OrbitType::AAB, 0.3956894730559419, 0.0, 0.009595471336070963},
                {OrbitType::AB0, 0.4783690288121502, 0.0, 0.009694996361663028}
            };

        case 146:
            // Degree 19.
            return {
                {OrbitType::A1, 0.0, 0.0, 0.0005996313688621381},
                {OrbitType::A2, 0.0, 0.0, 0.007372999718620756},
                {OrbitType::A3, 0.0, 0.0, 0.007210515360144488},
                {OrbitType::AAB, 0.6764410400114264, 0.0, 0.007116355493117555},
                {OrbitType::AAB, 0.4174961227965453, 0.0, 0.006753829486314477},
                {OrbitType::AAB, 0.1574676672039082, 0.0, 0.007574394159054034},
                {OrbitType::ABC, 0.1403553811713183, 0.4493328323269557, 0.006991087353303262}
            };

        case 170:
            // Degree 21.
            return {
                {OrbitType::A1, 0.0, 0.0, 0.005544842902037365},
                {OrbitType::A2, 0.0, 0.0, 0.006071332770670752},
                {OrbitType::A3, 0.0, 0.0, 0.006383674773515093},
                {OrbitType::AAB, 0.2551252621114134, 0.0, 0.005183387587747790},
                {OrbitType::AAB, 0.6743601460362766, 0.0, 0.006317929009813725},
                {OrbitType::AAB, 0.4318910696719410, 0.0, 0.006201670006589077},
                {OrbitType::AB0, 0.2613931360335988, 0.0, 0.005477143385137348},
                {OrbitType::ABC, 0.4990453161796037, 0.1446630744325115, 0.005968383987681156}
            };

        case 194:
            // Degree 23.
            return {
                {OrbitType::A1, 0.0, 0.0, 0.001782340447244611},
                {OrbitType::A2, 0.0, 0.0, 0.005716905949977102},
                {OrbitType::A3, 0.0, 0.0, 0.005573383178848738},
                {OrbitType::AAB, 0.6712973442695226, 0.0, 0.005608704082587997},
                {OrbitType::AAB, 0.2892465627575439, 0.0, 0.005158237711805383},
                {OrbitType::AAB, 0.4446933178717437, 0.0, 0.005518771467273614},
                {OrbitType::AAB, 0.1299335447650067, 0.0, 0.004106777028169394},
                {OrbitType::AB0, 0.3457702197611283, 0.0, 0.005051846064614808},
                {OrbitType::ABC, 0.1590417105383530, 0.8360360154824589, 0.005530248916233094}
            };

        case 230:
            // Degree 25.
            // Contains a negative A1 weight.
            return {
                {OrbitType::A1, 0.0, 0.0, -0.05522639919727325},
                {OrbitType::A3, 0.0, 0.0, 0.004450274607445226},
                {OrbitType::AAB, 0.4492044687397611, 0.0, 0.004496841067921404},
                {OrbitType::AAB, 0.2520419490210201, 0.0, 0.005049153450478750},
                {OrbitType::AAB, 0.6981906658447242, 0.0, 0.003976408018051883},
                {OrbitType::AAB, 0.6587405243460960, 0.0, 0.004401400650381014},
                {OrbitType::AAB, 0.04038544050097660, 0.0, 0.01724544350544401},
                {OrbitType::AB0, 0.5823842309715585, 0.0, 0.004231083095357343},
                {OrbitType::AB0, 0.3545877390518688, 0.0, 0.005198069864064399},
                {OrbitType::ABC, 0.2272181808998187, 0.4864661535886647, 0.004695720972568883}
            };

        case 266:
            // Degree 27.
            // Contains negative A1 and A2 weights.
            return {
                {OrbitType::A1, 0.0, 0.0, -0.001313769127326952},
                {OrbitType::A2, 0.0, 0.0, -0.002522728704859336},
                {OrbitType::A3, 0.0, 0.0, 0.004186853881700583},
                {OrbitType::AAB, 0.7039373391585475, 0.0, 0.005315167977810885},
                {OrbitType::AAB, 0.1012526248572414, 0.0, 0.004047142377086219},
                {OrbitType::AAB, 0.4647448726420539, 0.0, 0.004112482394406990},
                {OrbitType::AAB, 0.3277420654971629, 0.0, 0.003595584899758782},
                {OrbitType::AAB, 0.6620338663699974, 0.0, 0.004256131351428158},
                {OrbitType::AB0, 0.8506508083520399, 0.0, 0.004229582700647240},
                {OrbitType::ABC, 0.3233484542692899, 0.1153112011009701, 0.004080914225780505},
                {OrbitType::ABC, 0.2314790158712601, 0.5244939240922365, 0.004071467593830964}
            };

        case 302:
            // Degree 29.
            return {
                {OrbitType::A1, 0.0, 0.0, 0.0008545911725128148},
                {OrbitType::A3, 0.0, 0.0, 0.003599119285025571},
                {OrbitType::AAB, 0.3515640345570105, 0.0, 0.003449788424305883},
                {OrbitType::AAB, 0.6566329410219612, 0.0, 0.003604822601419882},
                {OrbitType::AAB, 0.4729054132581005, 0.0, 0.003576729661743367},
                {OrbitType::AAB, 0.09618308522614784, 0.0, 0.002352101413689164},
                {OrbitType::AAB, 0.2219645236294178, 0.0, 0.003108953122413675},
                {OrbitType::AAB, 0.7011766416089545, 0.0, 0.003650045807677255},
                {OrbitType::AB0, 0.2644152887060663, 0.0, 0.002982344963171804},
                {OrbitType::AB0, 0.5718955891878961, 0.0, 0.003600820932216460},
                {OrbitType::ABC, 0.2510034751770465, 0.8000727494073952, 0.003571540554273387},
                {OrbitType::ABC, 0.1233548532583327, 0.4127724083168531, 0.003392312205006170}
            };

        default:
            throw std::runtime_error(
                "unsupported Lebedev grid size: " + std::to_string(n_points) +
                ". Supported sizes are 6, 14, 26, 38, 50, 74, 86, 110, "
                "146, 170, 194, 230, 266, 302."
            );
    }
}

inline int lebedev_degree(int n_points) {
    switch (n_points) {
        case 6: return 3;
        case 14: return 5;
        case 26: return 7;
        case 38: return 9;
        case 50: return 11;
        case 74: return 13;
        case 86: return 15;
        case 110: return 17;
        case 146: return 19;
        case 170: return 21;
        case 194: return 23;
        case 230: return 25;
        case 266: return 27;
        case 302: return 29;

        // Common later values, to be enabled after adding tables:       
        // case 350: return 31;
        // case 434: return 35;
        // case 590: return 41;
        // case 770: return 47;
        // case 974: return 53;
        // case 1202: return 59;
        // case 1454: return 65;
        // case 1730: return 71;
        // case 2030: return 77;
        // case 2354: return 83;
        // case 2702: return 89;
        // case 3074: return 95;
        // case 3470: return 101;
        // case 3890: return 107;
        // case 4334: return 113;
        // case 4802: return 119;
        // case 5294: return 125;
        // case 5810: return 131;

        default:
            throw std::runtime_error(
                "unknown Lebedev degree for grid size: " + std::to_string(n_points)
            );
    }
}

inline std::vector<LebedevPoint> make_lebedev_grid(int n_points,
                                                   bool weights_sum_to_four_pi = true) {
    std::vector<LebedevPoint> points;

    const auto terms = rule_terms(n_points);

    for (const auto& term : terms) {
        add_rule_term(points, term);
    }

    if (static_cast<int>(points.size()) != n_points) {
        throw std::runtime_error(
            "Lebedev grid generated " + std::to_string(points.size()) +
            " points, but expected " + std::to_string(n_points)
        );
    }

    double wsum = 0.0;
    for (const auto& p : points) {
        const double norm = p.u.norm();
        if (std::abs(norm - 1.0) > 1.0e-12) {
            throw std::runtime_error("Lebedev point is not on the unit sphere");
        }
        wsum += p.w;
    }

    if (std::abs(wsum - 1.0) > 1.0e-12) {
        throw std::runtime_error(
            "Lebedev normalized weights do not sum to 1. Sum = " +
            std::to_string(wsum)
        );
    }

    if (weights_sum_to_four_pi) {
        const double factor = 4.0 * pi();
        for (auto& p : points) {
            p.w *= factor;
        }
    }

    return points;
}

inline bool is_supported(int n_points) {
    switch (n_points) {
        case 6:
        case 14:
        case 26:
        case 38:
        case 50:
        case 74:
        case 86:
        case 110:
        case 146:
        case 170:
        case 194:
        case 230:
        case 266:
        case 302:
            return true;
        default:
            return false;
    }
}

} // namespace lebedev
} // namespace dft