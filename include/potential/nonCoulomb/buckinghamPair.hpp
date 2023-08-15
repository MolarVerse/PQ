#ifndef _BUCKINGHAM_PAIR_HPP_

#define _BUCKINGHAM_PAIR_HPP_

#include "nonCoulombPair.hpp"

namespace potential
{
    class BuckinghamPair;
}   // namespace potential

/**
 * @class BuckinghamPair
 *
 * @brief inherits from NonCoulombPair represents a pair of Buckingham types
 *
 */
class potential::BuckinghamPair : public potential::NonCoulombPair
{
  private:
    double _a;
    double _dRho;
    double _c6;

  public:
    explicit BuckinghamPair(const size_t vanDerWaalsType1,
                            const size_t vanDerWaalsType2,
                            const double cutOff,
                            const double a,
                            const double dRho,
                            const double c6)
        : NonCoulombPair(vanDerWaalsType1, vanDerWaalsType2, cutOff), _a(a), _dRho(dRho), _c6(c6){};

    explicit BuckinghamPair(const double cutOff, const double a, const double dRho, const double c6)
        : NonCoulombPair(cutOff), _a(a), _dRho(dRho), _c6(c6){};

    explicit BuckinghamPair(const double cutOff,
                            const double energyCutoff,
                            const double forceCutoff,
                            const double a,
                            const double dRho,
                            const double c6)
        : NonCoulombPair(cutOff, energyCutoff, forceCutoff), _a(a), _dRho(dRho), _c6(c6){};

    [[nodiscard]] bool operator==(const BuckinghamPair &other) const;

    [[nodiscard]] std::pair<double, double> calculateEnergyAndForce(const double distance) const override;

    [[nodiscard]] double getA() const { return _a; }
    [[nodiscard]] double getDRho() const { return _dRho; }
    [[nodiscard]] double getC6() const { return _c6; }
};

#endif   // _BUCKINGHAM_PAIR_HPP_