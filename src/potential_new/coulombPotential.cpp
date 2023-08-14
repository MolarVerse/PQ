#include "coulombPotential.hpp"

#include <cmath>

using namespace potential_new;

CoulombPotential::CoulombPotential(const double coulombRadiusCutOff)
{
    _coulombRadiusCutOff = coulombRadiusCutOff;
    _coulombEnergyCutOff = 1 / _coulombRadiusCutOff;
    _coulombForceCutOff  = 1 / (_coulombRadiusCutOff * _coulombRadiusCutOff);
}

CoulombWolf::CoulombWolf(const double coulombRadiusCutOff, const double kappa) : CoulombPotential(coulombRadiusCutOff)
{
    _kappa          = kappa;
    _wolfParameter1 = ::erfc(_kappa * coulombRadiusCutOff) / coulombRadiusCutOff;
    _wolfParameter2 = 2.0 * _kappa / ::sqrt(M_PI);
    _wolfParameter3 = _wolfParameter1 / coulombRadiusCutOff +
                      _wolfParameter2 * ::exp(-_kappa * _kappa * coulombRadiusCutOff * coulombRadiusCutOff) / coulombRadiusCutOff;
}

[[nodiscard]] std::pair<double, double> CoulombShiftedPotential::calculateEnergyAndForce(const double distance) const
{
    auto energy = (1 / distance) - _coulombEnergyCutOff - _coulombForceCutOff * (_coulombRadiusCutOff - distance);
    auto force  = (1 / (distance * distance)) - _coulombForceCutOff;

    energy *= _coulombPreFactor;
    force  *= _coulombPreFactor;

    return {energy, force};
}

[[nodiscard]] std::pair<double, double> CoulombWolf::calculateEnergyAndForce(const double distance) const
{

    const auto kappaDistance = _kappa * distance;
    const auto erfcFactor    = ::erfc(kappaDistance);

    auto energy = erfcFactor / distance - _wolfParameter1 + _wolfParameter3 * (distance - _coulombRadiusCutOff);
    auto force =
        erfcFactor / (distance * distance) + _wolfParameter2 * ::exp(-kappaDistance * kappaDistance) / distance - _wolfParameter3;

    energy *= _coulombPreFactor;
    force  *= _coulombPreFactor;

    return {energy, force};
}

std::pair<double, double>
GuffCoulombShiftedPotential::calculate(const std::vector<size_t> &molAtomVdwIndices, const double distance, const double)
{
    auto molType1  = getMolType1(molAtomVdwIndices);
    auto molType2  = getMolType2(molAtomVdwIndices);
    auto atomType1 = getAtomType1(molAtomVdwIndices);
    auto atomType2 = getAtomType2(molAtomVdwIndices);

    _coulombPreFactor = _guffCoulombCoefficients[molType1 - 1][molType2 - 1][atomType1][atomType2];

    return calculateEnergyAndForce(distance);
}

std::pair<double, double>
GuffCoulombWolf::calculate(const std::vector<size_t> &molAtomVdwIndices, const double distance, const double)
{
    auto molType1  = getMolType1(molAtomVdwIndices);
    auto molType2  = getMolType2(molAtomVdwIndices);
    auto atomType1 = getAtomType1(molAtomVdwIndices);
    auto atomType2 = getAtomType2(molAtomVdwIndices);

    _coulombPreFactor = _guffCoulombCoefficients[molType1][molType2][atomType1][atomType2];

    return calculateEnergyAndForce(distance);
}

std::pair<double, double>
ForceFieldShiftedPotential::calculate(const std::vector<size_t> &, const double distance, const double coulombPreFactor)
{
    _coulombPreFactor = coulombPreFactor;

    return calculateEnergyAndForce(distance);
}

std::pair<double, double>
ForceFieldWolf::calculate(const std::vector<size_t> &, const double distance, const double coulombPreFactor)
{
    _coulombPreFactor = coulombPreFactor;

    return calculateEnergyAndForce(distance);
}