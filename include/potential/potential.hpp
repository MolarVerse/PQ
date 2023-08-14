#ifndef _POTENTIAL_HPP_

#define _POTENTIAL_HPP_

#include "celllist.hpp"
#include "coulombPotential.hpp"
#include "nonCoulombPotential.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"

namespace potential
{
    class Potential;
    class PotentialBruteForce;
    class PotentialCellList;
}   // namespace potential

class potential::Potential
{
  protected:
    std::unique_ptr<CoulombPotential>    _coulombPotential;
    std::unique_ptr<NonCoulombPotential> _nonCoulombPotential;

  public:
    virtual ~Potential() = default;

    std::pair<double, double> calculateSingleInteraction(
        const linearAlgebra::Vec3D &, simulationBox::Molecule &, simulationBox::Molecule &, const size_t, const size_t);
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

class potential::PotentialBruteForce : public potential::Potential
{
  public:
    void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) override;
};

class potential::PotentialCellList : public potential::Potential
{
  public:
    void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) override;
};

#endif   // _POTENTIAL_HPP_