#ifndef _POTENTIAL_NEW_HPP_   // TODO: refactor this to _POTENTIAL_HPP_

#define _POTENTIAL_NEW_HPP_

#include "celllist.hpp"
#include "coulombPotential.hpp"
#include "nonCoulombPotential.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"

namespace potential_new
{
    class Potential;
    class PotentialBruteForce;
    class PotentialCellList;
}   // namespace potential_new

class potential_new::Potential
{
  protected:
    std::unique_ptr<CoulombPotential>    _coulombPotential;
    std::unique_ptr<NonCoulombPotential> _nonCoulombPotential;

  public:
    virtual ~Potential() = default;

    virtual void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) = 0;

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

    [[nodiscard]] CoulombPotential    &getCoulombPotential() const { return *_coulombPotential; }
    [[nodiscard]] NonCoulombPotential &getNonCoulombPotential() const { return *_nonCoulombPotential; }
};

class potential_new::PotentialBruteForce : public potential_new::Potential
{
  public:
    void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) override;
};

class potential_new::PotentialCellList : public potential_new::Potential
{
  public:
    void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) override;
};

#endif   // _POTENTIAL_NEW_HPP_