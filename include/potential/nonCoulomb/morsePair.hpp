#ifndef _MORSE_PAIR_HPP_

#define _MORSE_PAIR_HPP_

#include "nonCoulombPair.hpp"

namespace potential
{
    class MorsePair;
}   // namespace potential

/**
 * @class MorsePair
 *
 * @brief inherits from NonCoulombPair represents a pair of Morse types
 *
 */
class potential::MorsePair : public potential::NonCoulombPair
{
  private:
    double _dissociationEnergy;
    double _wellWidth;
    double _equilibriumDistance;

  public:
    explicit MorsePair(const size_t vanDerWaalsType1,
                       const size_t vanDerWaalsType2,
                       const double cutOff,
                       const double dissociationEnergy,
                       const double wellWidth,
                       const double equilibriumDistance)
        : NonCoulombPair(vanDerWaalsType1, vanDerWaalsType2, cutOff), _dissociationEnergy(dissociationEnergy),
          _wellWidth(wellWidth), _equilibriumDistance(equilibriumDistance){};

    explicit MorsePair(const double cutOff,
                       const double dissociationEnergy,
                       const double wellWidth,
                       const double equilibriumDistance)
        : NonCoulombPair(cutOff), _dissociationEnergy(dissociationEnergy), _wellWidth(wellWidth),
          _equilibriumDistance(equilibriumDistance){};

    explicit MorsePair(const double cutOff,
                       const double energyCutoff,
                       const double forceCutoff,
                       const double dissociationEnergy,
                       const double wellWidth,
                       const double equilibriumDistance)
        : NonCoulombPair(cutOff, energyCutoff, forceCutoff), _dissociationEnergy(dissociationEnergy), _wellWidth(wellWidth),
          _equilibriumDistance(equilibriumDistance){};

    [[nodiscard]] bool operator==(const MorsePair &other) const;

    [[nodiscard]] std::pair<double, double> calculateEnergyAndForce(const double distance) const override;

    [[nodiscard]] double getDissociationEnergy() const { return _dissociationEnergy; }
    [[nodiscard]] double getWellWidth() const { return _wellWidth; }
    [[nodiscard]] double getEquilibriumDistance() const { return _equilibriumDistance; }
};

#endif   // _MORSE_PAIR_HPP_