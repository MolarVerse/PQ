#ifndef _POTENTIAL_HPP_

#define _POTENTIAL_HPP_

#include "celllist.hpp"
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
  public:
    virtual void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) = 0;
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

#endif   // _POTENTIAL_HPP_