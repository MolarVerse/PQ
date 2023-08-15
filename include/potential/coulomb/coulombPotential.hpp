#ifndef _COULOMB_POTENTIAL_HPP_

#define _COULOMB_POTENTIAL_HPP_

#include "defaults.hpp"

#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

namespace potential
{
    class CoulombPotential;
}   // namespace potential

using c_ul     = const size_t;
using vector4d = std::vector<std::vector<std::vector<std::vector<double>>>>;

/**
 * @class CoulombPotential
 *
 * @brief CoulombPotential is a base class for all coulomb type potentials
 *
 * @details it inherits in general the coulombRadiusCutOff, coulombEnergyCutOff and coulombForceCutOff to all coulomb type
 * potentials
 *
 * @details available for:
 * - shifted potential
 * - wolf summation as long range correction
 *
 */
class potential::CoulombPotential
{
  protected:
    inline static double _coulombRadiusCutOff = defaults::_COULOMB_CUT_OFF_DEFAULT_;
    inline static double _coulombEnergyCutOff = 1 / _coulombRadiusCutOff;
    inline static double _coulombForceCutOff  = 1 / (_coulombRadiusCutOff * _coulombRadiusCutOff);
    double               _coulombPreFactor;

    vector4d _guffCoulombCoefficients;

    std::function<void(const std::vector<size_t> &, const double)> _setCoulombPreFactor;

  public:
    virtual ~CoulombPotential() = default;
    explicit CoulombPotential(const double coulombRadiusCutOff);

    std::pair<double, double>         calculate(const std::vector<size_t> &, const double, const double);
    virtual std::pair<double, double> calculateEnergyAndForce(const double) const = 0;

    void setCoulombPreFactorToForceField();
    void setCoulombPreFactorToGuff();

    static void setCoulombRadiusCutOff(const double coulombRadiusCutOff);

    void resizeGuff(c_ul numberOfMoleculeTypes) { _guffCoulombCoefficients.resize(numberOfMoleculeTypes); }
    void resizeGuff(c_ul m1, c_ul numberOfMoleculeTypes) { _guffCoulombCoefficients[m1].resize(numberOfMoleculeTypes); }
    void resizeGuff(c_ul m1, c_ul m2, c_ul numberOfAtoms) { _guffCoulombCoefficients[m1][m2].resize(numberOfAtoms); }
    void resizeGuff(c_ul m1, c_ul m2, c_ul a1, c_ul numberOfAtoms) { _guffCoulombCoefficients[m1][m2][a1].resize(numberOfAtoms); }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    void setGuffCoulombCoefficient(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double guffCoulombCoefficient)
    {
        _guffCoulombCoefficients[m1 - 1][m2 - 1][a1][a2] = guffCoulombCoefficient;
    }

    static void setCoulombEnergyCutOff(const double coulombEnergyCutOff) { _coulombEnergyCutOff = coulombEnergyCutOff; }
    static void setCoulombForceCutOff(const double coulombForceCutOff) { _coulombForceCutOff = coulombForceCutOff; }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    [[nodiscard]] static double getCoulombRadiusCutOff() { return _coulombRadiusCutOff; }
    [[nodiscard]] static double getCoulombEnergyCutOff() { return _coulombEnergyCutOff; }
    [[nodiscard]] static double getCoulombForceCutOff() { return _coulombForceCutOff; }

    [[nodiscard]] size_t getMolType1(const std::vector<size_t> &indices) const { return indices[0]; }
    [[nodiscard]] size_t getMolType2(const std::vector<size_t> &indices) const { return indices[1]; }
    [[nodiscard]] size_t getAtomType1(const std::vector<size_t> &indices) const { return indices[2]; }
    [[nodiscard]] size_t getAtomType2(const std::vector<size_t> &indices) const { return indices[3]; }
};

#endif   // _COULOMB_POTENTIAL_HPP_