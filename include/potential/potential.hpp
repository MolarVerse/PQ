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
private:
    std::string _coulombType = "guff";
    std::string _nonCoulombType = "guff";

public:
    std::unique_ptr<CoulombPotential> _coulombPotential;
    std::unique_ptr<NonCoulombPotential> _nonCoulombPotential;

    virtual void calculateForces(SimulationBox &, OutputData &, CellList &) = 0;

    std::string getCoulombType() const { return _coulombType; };
    void setCoulombType(std::string_view coulombType) { _coulombType = coulombType; };

    std::string getNonCoulombType() const { return _nonCoulombType; };
    void setNonCoulombType(std::string_view nonCoulombType) { _nonCoulombType = nonCoulombType; };
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