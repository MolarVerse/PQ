#include "nonCoulombPotential.hpp"

#include <cmath>
#include <vector>

using namespace std;
using namespace potential;

void GuffBuckingham::calcNonCoulomb(const vector<double> &guffCoefficients,
                                    const double          rncCutoff,
                                    const double          distance,
                                    double               &energy,
                                    double               &force,
                                    const double          energy_cutoff,
                                    const double          force_cutoff) const
{
    const double c1 = guffCoefficients[0];
    const double c2 = guffCoefficients[1];
    const double c3 = guffCoefficients[2];

    const double helper = c1 * exp(distance * c2);

    const double distance_6 = distance * distance * distance * distance * distance * distance;
    const double helper_c3  = c3 / distance_6;

    energy  = helper + helper_c3;
    force  += -c2 * helper + 6 * helper_c3 / distance;

    energy -= energy_cutoff + force_cutoff * (rncCutoff - distance);
    force  -= force_cutoff;
}