#ifndef _NON_COULOMB_POTENTIAL_HPP_

#define _NON_COULOMB_POTENTIAL_HPP_

#include "matrix.hpp"
#include "nonCoulombPair.hpp"

#include <memory>
#include <vector>

namespace potential
{
    class NonCoulombPotential;
    class GuffNonCoulomb;
    class ForceFieldNonCoulomb;
}   // namespace potential

using c_ul                     = const size_t;
using vector4dNonCoulombicPair = std::vector<std::vector<std::vector<std::vector<std::shared_ptr<potential::NonCoulombPair>>>>>;

/**
 * @class NonCoulombPotential
 *
 * @brief NonCoulombPotential is a base class for all non coulomb potentials
 *
 */
class potential::NonCoulombPotential
{
  public:
    virtual ~NonCoulombPotential() = default;

    virtual void resizeGuff(c_ul){};
    virtual void resizeGuff(c_ul, c_ul){};
    virtual void resizeGuff(c_ul, c_ul, c_ul){};
    virtual void resizeGuff(c_ul, c_ul, c_ul, c_ul){};

    virtual void setGuffNonCoulombicPair(const std::vector<size_t> &, const std::shared_ptr<NonCoulombPair> &){};

    virtual std::shared_ptr<NonCoulombPair> getNonCoulombPair(const std::vector<size_t> &molAtomVdwIndices) = 0;

    [[nodiscard]] size_t getMolType1(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[0]; }
    [[nodiscard]] size_t getMolType2(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[1]; }
    [[nodiscard]] size_t getAtomType1(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[2]; }
    [[nodiscard]] size_t getAtomType2(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[3]; }
    [[nodiscard]] size_t getGlobalVdwType1(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[4]; }
    [[nodiscard]] size_t getGlobalVdwType2(const std::vector<size_t> &molAtomVdwIndices) const { return molAtomVdwIndices[5]; }
};

class potential::GuffNonCoulomb : public potential::NonCoulombPotential
{
  public:
    vector4dNonCoulombicPair _guffNonCoulombPairs;

    void resizeGuff(c_ul numberOfMoleculeTypes) override { _guffNonCoulombPairs.resize(numberOfMoleculeTypes); }
    void resizeGuff(c_ul m1, c_ul numberOfMoleculeTypes) override { _guffNonCoulombPairs[m1].resize(numberOfMoleculeTypes); }
    void resizeGuff(c_ul m1, c_ul m2, c_ul numberOfAtoms) override { _guffNonCoulombPairs[m1][m2].resize(numberOfAtoms); }
    void resizeGuff(c_ul m1, c_ul m2, c_ul a1, c_ul numberOfAtoms) override
    {
        _guffNonCoulombPairs[m1][m2][a1].resize(numberOfAtoms);
    }

    void setGuffNonCoulombicPair(const std::vector<size_t>             &molAtomVdwIndices,
                                 const std::shared_ptr<NonCoulombPair> &nonCoulombPair) override
    {
        _guffNonCoulombPairs[getMolType1(molAtomVdwIndices) - 1][getMolType2(molAtomVdwIndices) - 1]
                            [getAtomType1(molAtomVdwIndices)][getAtomType2(molAtomVdwIndices)] = nonCoulombPair;
    }

    std::shared_ptr<NonCoulombPair> getNonCoulombPair(const std::vector<size_t> &molAtomVdwIndices) override
    {
        return _guffNonCoulombPairs[getMolType1(molAtomVdwIndices) - 1][getMolType2(molAtomVdwIndices) - 1]
                                   [getAtomType1(molAtomVdwIndices)][getAtomType2(molAtomVdwIndices)];
    }
};

class potential::ForceFieldNonCoulomb : public potential::NonCoulombPotential
{
  public:
    linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>> _nonCoulombicPairsMatrix;

    std::shared_ptr<NonCoulombPair> getNonCoulombPair(const std::vector<size_t> &molAtomVdwIndices) override
    {
        return _nonCoulombicPairsMatrix[getGlobalVdwType1(molAtomVdwIndices)][getGlobalVdwType2(molAtomVdwIndices)];
    }
};

#endif   // _NON_COULOMB_POTENTIAL_HPP_