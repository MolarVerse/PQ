#include "nonCoulombPotential.hpp"

#include <cmath>
#include <vector>

using namespace std;
using namespace potential;

/**
 * @brief calculates the non-coulomb potential and its force for the general guff potential
 *
 * @param guffCoefficients
 * @param rncCutoff
 * @param distance
 * @param energy
 * @param force
 * @param energyCutoff
 * @param forceCutoff
 */
void GuffNonCoulomb::calcNonCoulomb(const vector<double> &guffCoefficients,
                                    const double          rncCutoff,
                                    const double          distance,
                                    double               &energy,
                                    double               &force,
                                    const double          energyCutoff,
                                    const double          forceCutoff) const
{
    const double c1 = guffCoefficients[0];
    const double n2 = guffCoefficients[1];
    const double c3 = guffCoefficients[2];
    const double n4 = guffCoefficients[3];

    const double distance_n2 = ::pow(distance, n2);
    const double distance_n4 = ::pow(distance, n4);

    auto energyLocal = c1 / distance_n2 + c3 / distance_n4;
    auto forceLocal  = n2 * c1 / (distance_n2 * distance) + n4 * c3 / (distance_n4 * distance);

    const double c5 = guffCoefficients[4];
    const double n6 = guffCoefficients[5];
    const double c7 = guffCoefficients[6];
    const double n8 = guffCoefficients[7];

    const double distance_n6 = ::pow(distance, n6);
    const double distance_n8 = ::pow(distance, n8);

    energyLocal += c5 / distance_n6 + c7 / distance_n8;
    forceLocal  += n6 * c5 / (distance_n6 * distance) + n8 * c7 / (distance_n8 * distance);

    const double c9     = guffCoefficients[8];
    const double cexp10 = guffCoefficients[9];
    const double rExp11 = guffCoefficients[10];

    double helper = ::exp(cexp10 * (distance - rExp11));

    energyLocal += c9 / (1 + helper);
    forceLocal  += c9 * cexp10 * helper / ((1 + helper) * (1 + helper));

    const double c12    = guffCoefficients[11];
    const double cexp13 = guffCoefficients[12];
    const double rExp14 = guffCoefficients[13];

    helper = ::exp(cexp13 * (distance - rExp14));

    energyLocal += c12 / (1 + helper);
    forceLocal  += c12 * cexp13 * helper / ((1 + helper) * (1 + helper));

    const double c15    = guffCoefficients[14];
    const double cexp16 = guffCoefficients[15];
    const double rExp17 = guffCoefficients[16];
    const double n18    = guffCoefficients[17];

    const double distance_n18 = ::pow(distance - rExp17, n18);
    helper                    = c15 * ::exp(cexp16 * distance_n18);

    energyLocal += helper;
    forceLocal  += -cexp16 * n18 * distance_n18 / (distance - rExp17) * helper;

    const double c19    = guffCoefficients[18];
    const double cexp20 = guffCoefficients[19];
    const double rExp21 = guffCoefficients[20];
    const double n22    = guffCoefficients[21];

    const double distance_n22 = ::pow(distance - rExp21, n22);
    helper                    = c19 * ::exp(cexp20 * distance_n22);

    energyLocal += helper;
    forceLocal  += -cexp20 * n22 * distance_n22 / (distance - rExp21) * helper;

    energyLocal += -energyCutoff - forceCutoff * (rncCutoff - distance);
    forceLocal  += -forceCutoff;

    energy  = energyLocal;
    force  += forceLocal;
}