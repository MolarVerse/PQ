#ifndef _NON_COULOMBIC_PAIR_HPP_

#define _NON_COULOMBIC_PAIR_HPP_

#include "mathUtilities.hpp"

#include <cstddef>
#include <vector>

namespace potential_new
{
    class NonCoulombicPair;
    class LennardJonesPair;
    class BuckinghamPair;
    class MorsePair;
}   // namespace potential_new

/**
 * @class NonCoulombicPair
 *
 * @brief base class representing a pair of non-coulombic types
 *
 * @details constructor with van der Waals types and cut-off radius is for force field parameters
 *         constructor with cut-off radius only is for guff representation
 *
 */
class potential_new::NonCoulombicPair
{
  protected:
    size_t _vanDerWaalsType1;
    size_t _vanDerWaalsType2;
    size_t _internalType1;
    size_t _internalType2;

    double _radialCutOff;
    double _energyCutOff = 0.0;
    double _forceCutOff  = 0.0;

  public:
    NonCoulombicPair(const size_t vanDerWaalsType1, const size_t vanDerWaalsType2, const double cutOff)
        : _vanDerWaalsType1(vanDerWaalsType1), _vanDerWaalsType2(vanDerWaalsType2), _radialCutOff(cutOff){};
    explicit NonCoulombicPair(const double cutOff) : _radialCutOff(cutOff){};
    virtual ~NonCoulombicPair() = default;

    // TODO: think of a way to generalize this also for guff routine
    [[nodiscard]] bool operator==(const NonCoulombicPair &other) const;

    virtual std::pair<double, double> calculateEnergyAndForce(const double distance) const = 0;

    void setInternalType1(const size_t internalType1) { _internalType1 = internalType1; }
    void setInternalType2(const size_t internalType2) { _internalType2 = internalType2; }
    void setEnergyCutOff(const double energyCutoff) { _energyCutOff = energyCutoff; }
    void setForceCutOff(const double forceCutoff) { _forceCutOff = forceCutoff; }

    [[nodiscard]] size_t getVanDerWaalsType1() const { return _vanDerWaalsType1; }
    [[nodiscard]] size_t getVanDerWaalsType2() const { return _vanDerWaalsType2; }
    [[nodiscard]] size_t getInternalType1() const { return _internalType1; }
    [[nodiscard]] size_t getInternalType2() const { return _internalType2; }
    [[nodiscard]] double getRadialCutOff() const { return _radialCutOff; }
};

/**
 * @class LennardJonesPair
 *
 * @brief represents a pair of Lennard-Jones types
 *
 */
class potential_new::LennardJonesPair : public potential_new::NonCoulombicPair
{
  private:
    double _c6;
    double _c12;

  public:
    LennardJonesPair(
        const size_t vanDerWaalsType1, const size_t vanDerWaalsType2, const double cutOff, const double c6, const double c12)
        : NonCoulombicPair(vanDerWaalsType1, vanDerWaalsType2, cutOff), _c6(c6), _c12(c12){};

    LennardJonesPair(const double cutOff, const double c6, const double c12) : NonCoulombicPair(cutOff), _c6(c6), _c12(c12){};

    [[nodiscard]] bool operator==(const LennardJonesPair &other) const;

    std::pair<double, double> calculateEnergyAndForce(const double distance) const override;

    [[nodiscard]] double getC6() const { return _c6; }
    [[nodiscard]] double getC12() const { return _c12; }
};

/**
 * @class BuckinghamPair
 *
 * @brief represents a pair of Buckingham types
 *
 */
class potential_new::BuckinghamPair : public potential_new::NonCoulombicPair
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
    BuckinghamPair(const double cutOff, const double a, const double dRho, const double c6)
        : NonCoulombicPair(cutOff), _a(a), _dRho(dRho), _c6(c6){};

    [[nodiscard]] bool operator==(const BuckinghamPair &other) const;

    std::pair<double, double> calculateEnergyAndForce(const double distance) const override;

    [[nodiscard]] double getA() const { return _a; }
    [[nodiscard]] double getDRho() const { return _dRho; }
    [[nodiscard]] double getC6() const { return _c6; }
};

/**
 * @class MorsePair
 *
 * @brief represents a pair of Morse types
 *
 */
class potential_new::MorsePair : public potential_new::NonCoulombicPair
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
    MorsePair(const double cutOff, const double dissociationEnergy, const double wellWidth, const double equilibriumDistance)
        : NonCoulombicPair(cutOff), _dissociationEnergy(dissociationEnergy), _wellWidth(wellWidth),
          _equilibriumDistance(equilibriumDistance){};

    [[nodiscard]] bool operator==(const MorsePair &other) const;

    std::pair<double, double> calculateEnergyAndForce(const double distance) const override;

    [[nodiscard]] double getDissociationEnergy() const { return _dissociationEnergy; }
    [[nodiscard]] double getWellWidth() const { return _wellWidth; }
    [[nodiscard]] double getEquilibriumDistance() const { return _equilibriumDistance; }
};

#endif   // _NON_COULOMBIC_PAIR_HPP_