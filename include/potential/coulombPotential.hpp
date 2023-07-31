#ifndef _COULOMB_POTENTIAL_HPP_

#define _COULOMB_POTENTIAL_HPP_

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
  public:
    virtual ~CoulombPotential() = default;
    virtual void calcCoulomb(const double, const double, const double, double &, double &, const double, const double) const = 0;
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
    void calcCoulomb(const double, const double, const double, double &, double &, const double, const double) const override;
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

  public:
    explicit GuffWolfCoulomb(const double wolfParameter) : _kappa(wolfParameter){};

    void calcCoulomb(const double, const double, const double, double &, double &, const double, const double) const override;
};

#endif   // _COULOMB_POTENTIAL_HPP_