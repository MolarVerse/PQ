#ifndef _COULOMB_POTENTIAL_HPP_

#define _COULOMB_POTENTIAL_HPP_

#include "defaults.hpp"

namespace potential
{
    class CoulombPotential;
    class GuffCoulomb;
    class GuffWolfCoulomb;
}   // namespace potential

/**
 * @class CoulombPotential
 *
 * @brief CoulombPotential is a base class for all coulomb potentials
 *
 */
class potential::CoulombPotential
{
  protected:
    double _coulombRadiusCutOff;

  public:
    CoulombPotential() : _coulombRadiusCutOff(defaults::_COULOMB_CUT_OFF_DEFAULT_) {}
    explicit CoulombPotential(const double coulombCutoff) : _coulombRadiusCutOff(coulombCutoff) {}

    virtual ~CoulombPotential()                                                                                = default;
    virtual void calcCoulomb(const double, const double, double &, double &, const double, const double) const = 0;
};

/**
 * @class GuffCoulomb
 *
 * @brief
 *  GuffCoulomb inherits CoulombPotential
 *  GuffCoulomb is a class for Guff potential
 *  it does not contain any long range corrections
 */
class potential::GuffCoulomb : public potential::CoulombPotential
{
  public:
    using CoulombPotential::CoulombPotential;
    void calcCoulomb(const double, const double, double &, double &, const double, const double) const override;
};

/**
 * @class GuffWolfCoulomb
 *
 * @brief
 * GuffWolfCoulomb inherits CoulombPotential
 * GuffWolfCoulomb is a class for Guff potential with Wolf long range correction
 *
 */
class potential::GuffWolfCoulomb : public potential::CoulombPotential
{
  private:
    double _kappa;
    double _wolfParameter1;
    double _wolfParameter2;
    double _wolfParameter3;

  public:
    GuffWolfCoulomb(const double coulombRadiusCutOff, const double wolfParameter);

    void calcCoulomb(const double, const double, double &, double &, const double, const double) const override;

    double getKappa() const { return _kappa; }
};

#endif   // _COULOMB_POTENTIAL_HPP_