#include "coulombPotential.hpp"

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
 * @brief sets the function pointer _setCoulombPreFactor to enable force field calculations
 *
 * @details here the coulombParameter is set to the product of partial charges divided by 4 * pi * epsilon_0
 *
 */
void CoulombPotential::setCoulombPreFactorToForceField()
{
    _setCoulombPreFactor = [this](const std::vector<size_t> &, const double coulombPreFactor)
    { _coulombPreFactor = coulombPreFactor; };
}

/**
 * @brief sets the function pointer _setCoulombPreFactor to enable guff calculations
 *
 * @details here the coulombParameter is selected from the guffCoulombCoefficients via the mol- and atom types
 *
 */
void CoulombPotential::setCoulombPreFactorToGuff()
{
    _setCoulombPreFactor = [this](const std::vector<size_t> &indices, const double)
    {
        _coulombPreFactor = _guffCoulombCoefficients[getMolType1(indices) - 1][getMolType2(indices) - 1][getAtomType1(indices)]
                                                    [getAtomType2(indices)];
    };
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

/**
 * @brief sets the coulombPreFactor and calculates the energy and force for a given distance by calling calculateEnergyAndForce of
 * the derived classes
 *
 * @note sonarlint states that this function should be const - maybe it miss-interprets the lambda function _setCoulombPreFactor
 *
 * @param distance
 * @return std::pair<double, double>
 */
[[nodiscard]] std::pair<double, double>
CoulombPotential::calculate(const std::vector<size_t> &indices, const double distance, const double preFactor)
{
    _setCoulombPreFactor(indices, preFactor);
    return calculateEnergyAndForce(distance);
}