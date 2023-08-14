#ifndef _GUFF_COULOMB_HPP_

#define _GUFF_COULOMB_HPP_

#include "coulombPotential.hpp"

namespace potential_new
{
    class GuffCoulomb;
}   // namespace potential_new

using c_ul     = const size_t;
using vector4d = std::vector<std::vector<std::vector<std::vector<double>>>>;

class potential_new::GuffCoulomb : public potential_new::CoulombPotential
{
  private:
    vector4d _guffCoulombCoefficients;

  public:
    using CoulombPotential::CoulombPotential;

    std::pair<double, double> calculate(std::vector<size_t> &, const double distance) override;
};

#endif   // _GUFF_COULOMB_HPP_