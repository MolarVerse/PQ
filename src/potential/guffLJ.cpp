#include "nonCoulombPotential.hpp"

#include <vector>

using namespace std;
using namespace potential;

/**
 * @brief calculates the non-coulomb potential and its force for the guff-lennard-jones potential
 *
 * @param guffCoefficients
 * @param rncCutoff
 * @param distance
 * @param energy
 * @param force
 * @param energy_cutoff
 * @param force_cutoff
 */
void GuffLennardJones::calcNonCoulomb(const vector<double> &guffCoefficients,
                                      const double          rncCutoff,
                                      const double          distance,
                                      double               &energy,
                                      double               &force,
                                      const double          energy_cutoff,
                                      const double          force_cutoff) const
{
    const double c6  = guffCoefficients[0];
    const double c12 = guffCoefficients[2];

    const double distance_6  = distance * distance * distance * distance * distance * distance;
    const double distance_12 = distance_6 * distance_6;

    energy  = c12 / distance_12 + c6 / distance_6;
    force  += 12 * c12 / (distance_12 * distance) + 6 * c6 / (distance_6 * distance);

    energy -= energy_cutoff + force_cutoff * (rncCutoff - distance);
    force  -= force_cutoff;
}