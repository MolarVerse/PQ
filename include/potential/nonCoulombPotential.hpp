#ifndef _NON_COULOMB_POTENTIAL_H_

#define _NON_COULOMB_POTENTIAL_H_

#include <vector>

/**
 * @class NonCoulombPotential
 *
 * @brief NonCoulombPotential is a base class for all non coulomb potentials
 *
 */
class NonCoulombPotential
{
public:
    virtual ~NonCoulombPotential() = default;
    virtual void calcNonCoulomb(std::vector<double> &, double, double, double &, double &, double, double) const = 0;
};

/**
 * @class GuffNonCoulomb
 *
 * @brief
 * GuffNonCoulomb inherits NonCoulombPotential
 * GuffNonCoulomb is a class for the full Guff potential
 *
 */
class GuffNonCoulomb : public NonCoulombPotential
{
public:
    void calcNonCoulomb(std::vector<double> &, double, double, double &, double &, double, double) const override;
};

/**
 * @class GuffLJ
 *
 * @brief
 * GuffLJ inherits NonCoulombPotential
 * GuffLJ is a class for the Lennard-Jones part of the Guff potential
 * it uses only parameters 1(C6) and 3(C12) of the guffdat file
 */
class GuffLJ : public NonCoulombPotential
{
public:
    void calcNonCoulomb(std::vector<double> &, double, double, double &, double &, double, double) const override;
};

/**
 * @class GuffBuckingham
 *
 * @brief
 * GuffBuckingham inherits NonCoulombPotential
 * GuffBuckingham is a class for the Buckingham part of the Guff potential
 * it uses only parameters 1(A), 2(B) and 3(C) of the guffdat file
 */
class GuffBuckingham : public NonCoulombPotential
{
public:
    void calcNonCoulomb(std::vector<double> &, double, double, double &, double &, double, double) const override;
};

#endif // _NON_COULOMB_POTENTIAL_H_