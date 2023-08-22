#include "coulombPotential.hpp"

#include "constants.hpp"

using namespace potential;

/**
 * @brief Construct a new Coulomb Potential:: Coulomb Potential object
 *
 * @details the coulomb energy cutoff is set to 1 / coulombRadiusCutOff and the coulomb force cutoff is set to 1 /
 * (coulombRadiusCutOff * coulombRadiusCutOff) the coulomb pre factor is not included here but later in the
 * calculateEnergyAndForce function of the derived classes
 *
 * @param coulombRadiusCutOff
 */
CoulombPotential::CoulombPotential(const double coulombRadiusCutOff)
{
    _coulombRadiusCutOff = coulombRadiusCutOff;
    _coulombEnergyCutOff = 1 / _coulombRadiusCutOff;
    _coulombForceCutOff  = 1 / (_coulombRadiusCutOff * _coulombRadiusCutOff);
}

/**
 * @brief sets the coulombRadiusCutOff and calculates the energy and force cutoff - equivalent to the constructor
 *
 * @details coulombPreFactor is not included in the energy and force cutoff
 *
 * @param coulombRadiusCutOff
 */
void CoulombPotential::setCoulombRadiusCutOff(const double coulombRadiusCutOff)
{
    _coulombRadiusCutOff = coulombRadiusCutOff;
    _coulombEnergyCutOff = 1 / _coulombRadiusCutOff;
    _coulombForceCutOff  = 1 / (_coulombRadiusCutOff * _coulombRadiusCutOff);
}