#ifndef _COULOMB_WOLF_HPP_

#define _COULOMB_WOLF_HPP_

#include "coulombPotential.hpp"

namespace potential
{
    class CoulombWolf;
}   // namespace potential

/**
 * @class CoulombWolf
 *
 * @brief
 * CoulombWolf inherits CoulombPotential
 * CoulombWolf is a class for the Coulomb potential with Wolf summation as long range correction
 *
 */
class potential::CoulombWolf : public potential::CoulombPotential
{
  protected:
    inline static double _kappa;
    inline static double _wolfParameter1;
    inline static double _wolfParameter2;
    inline static double _wolfParameter3;

  public:
    explicit CoulombWolf(const double coulombRadiusCutOff, const double kappa);

    [[nodiscard]] std::pair<double, double> calculateEnergyAndForce(const double) const override;

    static void setKappa(const double kappa) { _kappa = kappa; }
    static void setWolfParameter1(const double wolfParameter1) { _wolfParameter1 = wolfParameter1; }
    static void setWolfParameter2(const double wolfParameter2) { _wolfParameter2 = wolfParameter2; }
    static void setWolfParameter3(const double wolfParameter3) { _wolfParameter3 = wolfParameter3; }

    [[nodiscard]] double getKappa() const { return _kappa; }
    [[nodiscard]] double getWolfParameter1() const { return _wolfParameter1; }
    [[nodiscard]] double getWolfParameter2() const { return _wolfParameter2; }
    [[nodiscard]] double getWolfParameter3() const { return _wolfParameter3; }
};

#endif   // _COULOMB_WOLF_HPP_
