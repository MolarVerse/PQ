#ifndef _NON_COULOMBIC_PAIR_HPP_

#define _NON_COULOMBIC_PAIR_HPP_

#include "cstddef"

namespace forceField
{
    class NonCoulombicPair;
    class LennardJonesPair;
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
    NonCoulombicPair(size_t vanDerWaalsType1, size_t vanDerWaalsType2, double cutOff)
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
    double _C6;
    double _C12;

  public:
    LennardJonesPair(size_t vanDerWaalsType1, size_t vanDerWaalsType2, double cutOff, double C6, double C12)
        : NonCoulombicPair(vanDerWaalsType1, vanDerWaalsType2, cutOff), _C6(C6), _C12(C12){};

    double getC6() const { return _C6; }
    double getC12() const { return _C12; }
};

#endif   // _NON_COULOMBIC_PAIR_HPP_