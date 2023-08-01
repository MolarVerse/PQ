#ifndef _NON_COULOMB_POTENTIAL_HPP_

#define _NON_COULOMB_POTENTIAL_HPP_

#include <vector>

namespace potential
{
    class NonCoulombPotential;
    class GuffNonCoulomb;
    class GuffLennardJones;
    class GuffBuckingham;
}   // namespace potential

/**
 * @class NonCoulombPotential
 *
 * @brief NonCoulombPotential is a base class for all non coulomb potentials
 *
 */
class potential::NonCoulombPotential
{
  public:
    virtual ~NonCoulombPotential() = default;
    virtual void calcNonCoulomb(
        const std::vector<double> &, const double, const double, double &, double &, const double, const double) const = 0;
};

/**
 * @class GuffNonCoulomb
 *
 * @brief
 * GuffNonCoulomb inherits NonCoulombPotential
 * GuffNonCoulomb is a class for the full Guff potential
 *
 */
class potential::GuffNonCoulomb : public potential::NonCoulombPotential
{
  public:
    void calcNonCoulomb(
        const std::vector<double> &, const double, const double, double &, double &, const double, const double) const override;
};

/**
 * @class GuffLennardJones
 *
 * @brief
 * GuffLennardJones inherits NonCoulombPotential
 * GuffLennardJones is a class for the Lennard-Jones part of the Guff potential
 * it uses only parameters 1(C6) and 3(C12) of the guffdat file
 */
class potential::GuffLennardJones : public potential::NonCoulombPotential
{
  public:
    void calcNonCoulomb(
        const std::vector<double> &, const double, const double, double &, double &, const double, const double) const override;
};

/**
 * @class GuffBuckingham
 *
 * @brief
 * GuffBuckingham inherits NonCoulombPotential
 * GuffBuckingham is a class for the Buckingham part of the Guff potential
 * it uses only parameters 1(A), 2(B) and 3(C) of the guffdat file
 */
class potential::GuffBuckingham : public potential::NonCoulombPotential
{
  public:
    void calcNonCoulomb(
        const std::vector<double> &, const double, const double, double &, double &, const double, const double) const override;
};

#endif   // _NON_COULOMB_POTENTIAL_HPP_