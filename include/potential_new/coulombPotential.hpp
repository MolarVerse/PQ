#ifndef _COULOMB_POTENTIAL_NEW_HPP_   // TODO: refactor this to _COULOMB_POTENTIAL_HPP_

#define _COULOMB_POTENTIAL_NEW_HPP_

#include "defaults.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

namespace potential_new
{
    class CoulombPotential;

    class CoulombShiftedPotential;
    class CoulombWolf;
    class GuffCoulombShiftedPotential;
    class GuffCoulombWolf;
    class ForceFieldShiftedPotential;
    class ForceFieldWolf;
}   // namespace potential_new

using c_ul     = const size_t;
using vector4d = std::vector<std::vector<std::vector<std::vector<double>>>>;

class potential_new::CoulombPotential
{
  protected:
    inline static double _coulombRadiusCutOff;   // TODO: add default value here
    inline static double _coulombEnergyCutOff;   // TODO: add default value here
    inline static double _coulombForceCutOff;    // TODO: add default value here
    double               _coulombPreFactor;

  public:
    virtual ~CoulombPotential() = default;
    explicit CoulombPotential(const double coulombRadiusCutOff);

    virtual std::pair<double, double> calculate(const std::vector<size_t> &, const double, const double) = 0;
    virtual void                      resizeGuff(c_ul){};
    virtual void                      resizeGuff(c_ul, c_ul){};
    virtual void                      resizeGuff(c_ul, c_ul, c_ul){};
    virtual void                      resizeGuff(c_ul, c_ul, c_ul, c_ul){};
    virtual void                      setGuffCoulombCoefficient(c_ul, c_ul, c_ul, c_ul, const double){};

    static void setCoulombRadiusCutOff(const double coulombRadiusCutOff)
    {
        _coulombRadiusCutOff = coulombRadiusCutOff;
        _coulombEnergyCutOff = 1 / _coulombRadiusCutOff;
        _coulombForceCutOff  = 1 / (_coulombRadiusCutOff * _coulombRadiusCutOff);
    }
    static void setCoulombEnergyCutOff(const double coulombEnergyCutOff) { _coulombEnergyCutOff = coulombEnergyCutOff; }
    static void setCoulombForceCutOff(const double coulombForceCutOff) { _coulombForceCutOff = coulombForceCutOff; }

    [[nodiscard]] double getCoulombRadiusCutOff() const { return _coulombRadiusCutOff; }
    [[nodiscard]] double getCoulombEnergyCutOff() const { return _coulombEnergyCutOff; }
    [[nodiscard]] double getCoulombForceCutOff() const { return _coulombForceCutOff; }

    [[nodiscard]] size_t getMolType1(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[0]; }
    [[nodiscard]] size_t getMolType2(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[1]; }
    [[nodiscard]] size_t getAtomType1(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[2]; }
    [[nodiscard]] size_t getAtomType2(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[3]; }
};

class potential_new::CoulombShiftedPotential : public potential_new::CoulombPotential
{
  public:
    using CoulombPotential::CoulombPotential;

    std::pair<double, double> calculate(const std::vector<size_t> &, const double, const double) override = 0;

    [[nodiscard]] std::pair<double, double> calculateEnergyAndForce(const double) const;
};

class potential_new::CoulombWolf : public potential_new::CoulombPotential
{
  protected:
    inline static double _kappa;
    inline static double _wolfParameter1;
    inline static double _wolfParameter2;
    inline static double _wolfParameter3;

  public:
    explicit CoulombWolf(const double coulombRadiusCutOff, const double kappa);

    std::pair<double, double> calculate(const std::vector<size_t> &, const double, const double) override = 0;

    [[nodiscard]] std::pair<double, double> calculateEnergyAndForce(const double) const;   // TODO: implement

    static void setKappa(const double kappa) { _kappa = kappa; }
    static void setWolfParameter1(const double wolfParameter1) { _wolfParameter1 = wolfParameter1; }
    static void setWolfParameter2(const double wolfParameter2) { _wolfParameter2 = wolfParameter2; }
    static void setWolfParameter3(const double wolfParameter3) { _wolfParameter3 = wolfParameter3; }

    [[nodiscard]] double getKappa() const { return _kappa; }
    [[nodiscard]] double getWolfParameter1() const { return _wolfParameter1; }
    [[nodiscard]] double getWolfParameter2() const { return _wolfParameter2; }
    [[nodiscard]] double getWolfParameter3() const { return _wolfParameter3; }
};

class potential_new::GuffCoulombShiftedPotential : public potential_new::CoulombShiftedPotential
{
  private:
    vector4d _guffCoulombCoefficients;

  public:
    using CoulombShiftedPotential::CoulombShiftedPotential;

    std::pair<double, double> calculate(const std::vector<size_t> &, const double, const double) override;

    void resizeGuff(c_ul numberOfMoleculeTypes) override { _guffCoulombCoefficients.resize(numberOfMoleculeTypes); }
    void resizeGuff(c_ul m1, c_ul numberOfMoleculeTypes) override { _guffCoulombCoefficients[m1].resize(numberOfMoleculeTypes); }
    void resizeGuff(c_ul m1, c_ul m2, c_ul numberOfAtoms) override { _guffCoulombCoefficients[m1][m2].resize(numberOfAtoms); }
    void resizeGuff(c_ul m1, c_ul m2, c_ul a1, c_ul numberOfAtoms) override
    {
        _guffCoulombCoefficients[m1][m2][a1].resize(numberOfAtoms);
    }
    void setGuffCoulombCoefficient(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double guffCoulombCoefficient) override
    {
        _guffCoulombCoefficients[m1 - 1][m2 - 1][a1][a2] = guffCoulombCoefficient;
    }
};

class potential_new::GuffCoulombWolf : public potential_new::CoulombWolf
{
  private:
    vector4d _guffCoulombCoefficients;

  public:
    using CoulombWolf::CoulombWolf;

    std::pair<double, double> calculate(const std::vector<size_t> &, const double, const double) override;

    void resizeGuff(c_ul numberOfMoleculeTypes) override { _guffCoulombCoefficients.resize(numberOfMoleculeTypes); }
    void resizeGuff(c_ul m1, c_ul numberOfMoleculeTypes) override { _guffCoulombCoefficients[m1].resize(numberOfMoleculeTypes); }
    void resizeGuff(c_ul m1, c_ul m2, c_ul numberOfAtoms) override { _guffCoulombCoefficients[m1][m2].resize(numberOfAtoms); }
    void resizeGuff(c_ul m1, c_ul m2, c_ul a1, c_ul numberOfAtoms) override
    {
        _guffCoulombCoefficients[m1][m2][a1].resize(numberOfAtoms);
    }
    void setGuffCoulombCoefficient(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double guffCoulombCoefficient) override
    {
        _guffCoulombCoefficients[m1 - 1][m2 - 1][a1][a2] = guffCoulombCoefficient;
    }
};

class potential_new::ForceFieldShiftedPotential : public potential_new::CoulombShiftedPotential
{
  public:
    using CoulombShiftedPotential::CoulombShiftedPotential;

    std::pair<double, double> calculate(const std::vector<size_t> &, const double, const double) override;
};

class potential_new::ForceFieldWolf : public potential_new::CoulombWolf
{
  public:
    using CoulombWolf::CoulombWolf;

    std::pair<double, double> calculate(const std::vector<size_t> &, const double, const double) override;
};

#endif   // _COULOMB_POTENTIAL_NEW_HPP_