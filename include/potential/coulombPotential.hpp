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
    virtual void calcCoulomb(double, double, double, double &, double &, double, double force_cutof) const = 0;
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
    void calcCoulomb(double, double, double, double &, double &, double, double force_cutof) const override;
};

#endif // _COULOMB_POTENTIAL_H_