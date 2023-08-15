#ifndef _LENNARD_JONES_PAIR_HPP_

#define _LENNARD_JONES_PAIR_HPP_

#include "nonCoulombPair.hpp"

#include <cmath>

namespace potential
{
    class LennardJonesPair;
}   // namespace potential

/**
 * @class LennardJonesPair
 *
 * @brief inherits from NonCoulombPair and represents a pair of Lennard-Jones types
 *
 */
class potential::LennardJonesPair : public potential::NonCoulombPair
{
  private:
    double _c6;
    double _c12;

  public:
    explicit LennardJonesPair(
        const size_t vanDerWaalsType1, const size_t vanDerWaalsType2, const double cutOff, const double c6, const double c12)
        : NonCoulombPair(vanDerWaalsType1, vanDerWaalsType2, cutOff), _c6(c6), _c12(c12){};

    explicit LennardJonesPair(const double cutOff, const double c6, const double c12)
        : NonCoulombPair(cutOff), _c6(c6), _c12(c12){};

    explicit LennardJonesPair(
        const double cutOff, const double energyCutoff, const double forceCutoff, const double c6, const double c12)
        : NonCoulombPair(cutOff, energyCutoff, forceCutoff), _c6(c6), _c12(c12){};

    [[nodiscard]] bool operator==(const LennardJonesPair &other) const;

    [[nodiscard]] std::pair<double, double> calculateEnergyAndForce(const double distance) const override;

    [[nodiscard]] double getC6() const { return _c6; }
    [[nodiscard]] double getC12() const { return _c12; }
};

#endif   // _LENNARD_JONES_PAIR_HPP_