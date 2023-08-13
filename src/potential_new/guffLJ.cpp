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
 * @param energyCutoff
 * @param forceCutoff
 */
void GuffLennardJones::calcNonCoulomb(const vector<double> &guffCoefficients,
                                      const double          rncCutoff,
                                      const double          distance,
                                      double               &energy,
                                      double               &force,
                                      const double          energyCutoff,
                                      const double          forceCutoff) const
{
    const double c6  = guffCoefficients[0];
    const double c12 = guffCoefficients[2];

    const double distance_6  = distance * distance * distance * distance * distance * distance;
    const double distance_12 = distance_6 * distance_6;

    energy  = c12 / distance_12 + c6 / distance_6;
    force  += 12 * c12 / (distance_12 * distance) + 6 * c6 / (distance_6 * distance);

    energy -= energyCutoff + forceCutoff * (rncCutoff - distance);
    force  -= forceCutoff;
}