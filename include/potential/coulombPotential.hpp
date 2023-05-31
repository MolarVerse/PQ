#ifndef _COULOMB_POTENTIAL_H_

#define _COULOMB_POTENTIAL_H_

/**
 * @class CoulombPotential
 *
 * @brief CoulombPotential is a base class for all coulomb potentials
 *
 */
class CoulombPotential
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
class GuffCoulomb : public CoulombPotential
{
public:
    void calcCoulomb(const double, const double, const double, double &, double &, const double, const double) const override;
};

#endif // _COULOMB_POTENTIAL_H_