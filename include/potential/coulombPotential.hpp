#ifndef _COULOMB_POTENTIAL_H_

#define _COULOMB_POTENTIAL_H_

class CoulombPotential
{
public:
    virtual void calcCoulomb(double, double, double, double &, double &, double, double force_cutof) const = 0;
};

class GuffCoulomb : public CoulombPotential
{
public:
    void calcCoulomb(double, double, double, double &, double &, double, double force_cutof) const override;
};

#endif // _COULOMB_POTENTIAL_H_