#ifndef _POTENTIAL_H_

#define _POTENTIAL_H_

#include <vector>
#include <memory>

#include "coulombPotential.hpp"
#include "nonCoulombPotential.hpp"
#include "simulationBox.hpp"
#include "outputData.hpp"
#include "celllist.hpp"

class Potential
{
public:
    std::unique_ptr<CoulombPotential> _coulombPotential = std::make_unique<GuffCoulomb>();
    std::unique_ptr<NonCoulombPotential> _nonCoulombPotential = std::make_unique<GuffNonCoulomb>();

    virtual void calculateForces(SimulationBox &, OutputData &, CellList &) = 0;
};

class PotentialBruteForce : public Potential
{
public:
    void calculateForces(SimulationBox &, OutputData &, CellList &) override;
};

class PotentialCellList : public Potential
{
public:
    void calculateForces(SimulationBox &, OutputData &, CellList &) override;
};

#endif // _POTENTIAL_H_