#ifndef _NON_COULOMBIC_PAIR_HPP_

#define _NON_COULOMBIC_PAIR_HPP_

#include <cstddef>
#include <vector>

namespace forceField
{
    class NonCoulombicPair;
    class LennardJonesPair;
    class BuckinghamPair;
    class MorsePair;
}   // namespace forceField

/**
 * @class NonCoulombicPair
 *
 * @brief base class representing a pair of non-coulombic types
 *
 */
class forceField::NonCoulombicPair
{
  protected:
    size_t _vanDerWaalsType1;
    size_t _vanDerWaalsType2;

    double _cutOff;

  public:
    NonCoulombicPair(const size_t vanDerWaalsType1, const size_t vanDerWaalsType2, const double cutOff)
        : _vanDerWaalsType1(vanDerWaalsType1), _vanDerWaalsType2(vanDerWaalsType2), _cutOff(cutOff){};
    virtual ~NonCoulombicPair() = default;

    size_t getVanDerWaalsType1() const { return _vanDerWaalsType1; }
    size_t getVanDerWaalsType2() const { return _vanDerWaalsType2; }
    double getCutOff() const { return _cutOff; }
};

/**
 * @class LennardJonesPair
 *
 * @brief represents a pair of Lennard-Jones types
 *
 */
class forceField::LennardJonesPair : public forceField::NonCoulombicPair
{
  private:
    double _c6;
    double _c12;

  public:
    LennardJonesPair(
        const size_t vanDerWaalsType1, const size_t vanDerWaalsType2, const double cutOff, const double c6, const double c12)
        : NonCoulombicPair(vanDerWaalsType1, vanDerWaalsType2, cutOff), _c6(c6), _c12(c12){};

    double getC6() const { return _c6; }
    double getC12() const { return _c12; }
};

/**
 * @class BuckinghamPair
 *
 * @brief represents a pair of Buckingham types
 *
 */
class forceField::BuckinghamPair : public forceField::NonCoulombicPair
{
  private:
    double _a;
    double _dRho;
    double _c6;

  public:
    BuckinghamPair(const size_t vanDerWaalsType1,
                   const size_t vanDerWaalsType2,
                   const double cutOff,
                   const double a,
                   const double dRho,
                   const double c6)
        : NonCoulombicPair(vanDerWaalsType1, vanDerWaalsType2, cutOff), _a(a), _dRho(dRho), _c6(c6){};

    double getA() const { return _a; }
    double getDRho() const { return _dRho; }
    double getC6() const { return _c6; }
};

/**
 * @class MorsePair
 *
 * @brief represents a pair of Morse types
 *
 */
class forceField::MorsePair : public forceField::NonCoulombicPair
{
  private:
    double _dissociationEnergy;
    double _wellWidth;
    double _equilibriumDistance;

  public:
    MorsePair(const size_t vanDerWaalsType1,
              const size_t vanDerWaalsType2,
              const double cutOff,
              const double dissociationEnergy,
              const double wellWidth,
              const double equilibriumDistance)
        : NonCoulombicPair(vanDerWaalsType1, vanDerWaalsType2, cutOff), _dissociationEnergy(dissociationEnergy),
          _wellWidth(wellWidth), _equilibriumDistance(equilibriumDistance){};

    double getDissociationEnergy() const { return _dissociationEnergy; }
    double getWellWidth() const { return _wellWidth; }
    double getEquilibriumDistance() const { return _equilibriumDistance; }
};

#endif   // _NON_COULOMBIC_PAIR_HPP_