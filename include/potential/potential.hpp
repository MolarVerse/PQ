#ifndef _POTENTIAL_HPP_

#define _POTENTIAL_HPP_

#include "celllist.hpp"
#include "coulombPotential.hpp"
#include "matrix.hpp"
#include "nonCoulombPotential.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"

namespace potential
{
    class Potential;
    class PotentialBruteForce;
    class PotentialCellList;
    enum class NonCoulombType : size_t;
    enum class MixingRule : size_t;
}   // namespace potential

enum class potential::NonCoulombType : size_t
{
    LJ,
    LJ_9_12,   // at the momentum just dummy for testing not implemented yet
    BUCKINGHAM,
    MORSE,
    GUFF
};

enum class potential::MixingRule : size_t
{
    NONE
};

/**
 * @class Potential
 *
 * @brief base class for all potential routines
 *
 * @details
 * possible options:
 * - brute force
 * - cell list
 *
 * @note _nonCoulombicPairsVector is just a container to store the nonCoulombicPairs for later processing
 *
 */
class potential::Potential
{
  protected:
    NonCoulombType _nonCoulombType = NonCoulombType::LJ;   // LJ
    MixingRule     _mixingRule     = MixingRule::NONE;     // no mixing rule

    std::unique_ptr<CoulombPotential>                      _coulombPotential;
    std::unique_ptr<NonCoulombPotential>                   _nonCoulombPotential;
    std::vector<std::shared_ptr<NonCoulombPair>>           _nonCoulombicPairsVector;
    linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>> _nonCoulombicPairsMatrix;

  public:
    virtual ~Potential() = default;

    std::pair<double, double> calculateSingleInteraction(
        const linearAlgebra::Vec3D &, simulationBox::Molecule &, simulationBox::Molecule &, const size_t, const size_t);
    virtual void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) = 0;

    void addNonCoulombicPair(const std::shared_ptr<NonCoulombPair> &pair) { _nonCoulombicPairsVector.push_back(pair); }

    void determineInternalGlobalVdwTypes(const std::map<size_t, size_t> &);
    void fillDiagonalElementsOfNonCoulombicPairsMatrix(std::vector<std::shared_ptr<NonCoulombPair>> &);
    void fillNonDiagonalElementsOfNonCoulombicPairsMatrix();
    std::vector<std::shared_ptr<NonCoulombPair>>   getSelfInteractionNonCoulombicPairs() const;
    std::optional<std::shared_ptr<NonCoulombPair>> findNonCoulombicPairByInternalTypes(size_t, size_t) const;

    void initNonCoulombicPairsMatrix(const size_t n)
    {
        _nonCoulombicPairsMatrix = linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>>(n);
    }

    template <typename T>
    void makeCoulombPotential(T coulombPotential)
    {
        _coulombPotential = std::make_unique<T>(coulombPotential);
    }
    template <typename T>
    void makeNonCoulombPotential(T nonCoulombPotential)
    {
        _nonCoulombPotential = std::make_unique<T>(nonCoulombPotential);
    }

    void setNonCoulombType(const NonCoulombType nonCoulombType) { _nonCoulombType = nonCoulombType; }
    void setMixingRule(const MixingRule mixingRule) { _mixingRule = mixingRule; }
    void setNonCoulombicPairsVector(const std::vector<std::shared_ptr<NonCoulombPair>> &nonCoulombicPairsVector)
    {
        _nonCoulombicPairsVector = nonCoulombicPairsVector;
    }

    [[nodiscard]] NonCoulombType                                getNonCoulombType() const { return _nonCoulombType; }
    [[nodiscard]] MixingRule                                    getMixingRule() const { return _mixingRule; }
    [[nodiscard]] CoulombPotential                             &getCoulombPotential() const { return *_coulombPotential; }
    [[nodiscard]] NonCoulombPotential                          &getNonCoulombPotential() const { return *_nonCoulombPotential; }
    [[nodiscard]] std::vector<std::shared_ptr<NonCoulombPair>> &getNonCoulombicPairsVector() { return _nonCoulombicPairsVector; }

    [[nodiscard]] linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>> &getNonCoulombicPairsMatrix()
    {
        return _nonCoulombicPairsMatrix;
    }
};

/**
 * @class PotentialBruteForce
 *
 * @brief brute force implementation of the potential
 *
 */
class potential::PotentialBruteForce : public potential::Potential
{
  public:
    void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) override;
};

/**
 * @class PotentialCellList
 *
 * @brief cell list implementation of the potential
 *
 */
class potential::PotentialCellList : public potential::Potential
{
  public:
    void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) override;
};

#endif   // _POTENTIAL_HPP_