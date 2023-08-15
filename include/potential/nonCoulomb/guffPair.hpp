#ifndef _GUFF_PAIR_HPP_

#define _GUFF_PAIR_HPP_

#include "nonCoulombPair.hpp"

namespace potential
{
    class GuffPair;
}   // namespace potential

/**
 * @class GuffPair
 *
 * @brief inherits from NonCoulombPair represents a pair of Guff types (full guff formula)
 *
 * @note here the constructor including the van der Waals types is missing as this class is only used for guff potentials.
 * Therefore also the comparison operator == is missing.
 *
 */
class potential::GuffPair : public potential::NonCoulombPair
{
  private:
    std::vector<double> _coefficients;

  public:
    explicit GuffPair(const double cutOff, const std::vector<double> &coefficients)
        : NonCoulombPair(cutOff), _coefficients(coefficients){};

    explicit GuffPair(const double               cutOff,
                      const double               energyCutoff,
                      const double               forceCutoff,
                      const std::vector<double> &coefficients)
        : NonCoulombPair(cutOff, energyCutoff, forceCutoff), _coefficients(coefficients){};

    [[nodiscard]] std::pair<double, double> calculateEnergyAndForce(const double distance) const override;
};

#endif   // _GUFF_PAIR_HPP_