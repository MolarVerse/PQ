#ifndef _COULOMB_POTENTIAL_HPP_

#define _COULOMB_POTENTIAL_HPP_

#include "defaults.hpp"

#include <cstddef>
#include <vector>

namespace potential_new
{
    class CoulombPotential;
    class CoulombPotentialGuff;
    class CoulombPotentialForceField;
}   // namespace potential_new

class potential_new::CoulombPotential
{
  protected:
    double _coulombRadialCutOff;   // TODO: add default value here

  public:
    virtual ~CoulombPotential() = default;
    explicit CoulombPotential(const double coulombRadialCutOff) : _coulombRadialCutOff(coulombRadialCutOff) {}

    virtual std::pair<double, double> calculate(std::vector<size_t> &, const double distance) = 0;

    [[nodiscard]] size_t getMolType1(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[0]; }
    [[nodiscard]] size_t getMolType2(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[1]; }
    [[nodiscard]] size_t getAtomType1(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[2]; }
    [[nodiscard]] size_t getAtomType2(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[3]; }
    [[nodiscard]] size_t getVdwType1(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[4]; }
    [[nodiscard]] size_t getVdwType2(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[5]; }
};

using c_ul     = const size_t;
using vector4d = std::vector<std::vector<std::vector<std::vector<double>>>>;

class potential_new::CoulombPotentialGuff : public potential_new::CoulombPotential
{
  private:
    vector4d _guffCoulombCoefficients;

  public:
    using CoulombPotential::CoulombPotential;

    std::pair<double, double> calculate(std::vector<size_t> &, const double distance) override;
};

class potential_new::CoulombPotentialForceField : public potential_new::CoulombPotential
{
  public:
    using CoulombPotential::CoulombPotential;

    std::pair<double, double> calculate(std::vector<size_t> &, const double distance) override;
};

#endif   // _COULOMB_POTENTIAL_HPP_