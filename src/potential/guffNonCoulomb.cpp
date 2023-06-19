#include "nonCoulombPotential.hpp"

#include <cmath>
#include <vector>

using namespace std;
using namespace potential;

void GuffNonCoulomb::calcNonCoulomb(const vector<double> &guffCoefficients,
                                    const double          rncCutoff,
                                    const double          distance,
                                    double               &energy,
                                    double               &force,
                                    const double          energy_cutoff,
                                    const double          force_cutoff) const
{
    const double c1 = guffCoefficients[0];
    const double n2 = guffCoefficients[1];
    const double c3 = guffCoefficients[2];
    const double n4 = guffCoefficients[3];

    energy = c1 / pow(distance, n2) + c3 / pow(distance, n4);
    force += n2 * c1 / pow(distance, n2 + 1) + n4 * c3 / pow(distance, n4 + 1);

    const double c5 = guffCoefficients[4];
    const double n6 = guffCoefficients[5];
    const double c7 = guffCoefficients[6];
    const double n8 = guffCoefficients[7];

    energy += c5 / pow(distance, n6) + c7 / pow(distance, n8);
    force += n6 * c5 / pow(distance, n6 + 1) + n8 * c7 / pow(distance, n8 + 1);

    const double c9     = guffCoefficients[8];
    const double cexp10 = guffCoefficients[9];
    const double rexp11 = guffCoefficients[10];

    double helper = exp(cexp10 * (distance - rexp11));

    energy += c9 / (1 + helper);
    force += c9 * cexp10 * helper / ((1 + helper) * (1 + helper));

    const double c12    = guffCoefficients[11];
    const double cexp13 = guffCoefficients[12];
    const double rexp14 = guffCoefficients[13];

    helper = exp(cexp13 * (distance - rexp14));

    energy += c12 / (1 + helper);
    force += c12 * cexp13 * helper / ((1 + helper) * (1 + helper));

    const double c15    = guffCoefficients[14];
    const double cexp16 = guffCoefficients[15];
    const double rexp17 = guffCoefficients[16];
    const double n18    = guffCoefficients[17];

    helper = c15 * exp(cexp16 * pow((distance - rexp17), n18));

    energy += helper;
    force += -cexp16 * n18 * pow((distance - rexp17), n18 - 1) * helper;

    const double c19    = guffCoefficients[18];
    const double cexp20 = guffCoefficients[19];
    const double rexp21 = guffCoefficients[20];
    const double n22    = guffCoefficients[21];

    helper = c19 * exp(cexp20 * pow((distance - rexp21), n22));

    energy += helper;
    force += -cexp20 * n22 * pow((distance - rexp21), n22 - 1) * helper;

    energy += -energy_cutoff - force_cutoff * (rncCutoff - distance);
    force += -force_cutoff;
}