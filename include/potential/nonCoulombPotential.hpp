#ifndef _NON_COULOMB_POTENTIAL_H_

#define _NON_COULOMB_POTENTIAL_H_

#include <vector>

class NonCoulombPotential
{
public:
    virtual void calcNonCoulomb(std::vector<double> &, double, double, double &, double &, double, double) const = 0;
};

class GuffNonCoulomb : public NonCoulombPotential
{
public:
    void calcNonCoulomb(std::vector<double> &, double, double, double &, double &, double, double) const override;
};

class GuffLJ : public NonCoulombPotential
{
public:
    void calcNonCoulomb(std::vector<double> &, double, double, double &, double &, double, double) const override;
};

class GuffBuckingham : public NonCoulombPotential
{
public:
    void calcNonCoulomb(std::vector<double> &, double, double, double &, double &, double, double) const override;
};

#endif // _NON_COULOMB_POTENTIAL_H_