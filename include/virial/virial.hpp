#ifndef _VIRIAL_H_

#define _VIRIAL_H_

#include "simulationBox.hpp"
#include "physicalData.hpp"

#include <vector>

class Virial
{
protected:
    std::vector<double> _virial = {0.0, 0.0, 0.0};

public:
    virtual ~Virial() = default;

    virtual void computeVirial(SimulationBox &simulationBox, PhysicalData &physicalData) = 0;

    // standard getter and setters
    std::vector<double> getVirial() { return _virial; };
    void setVirial(std::vector<double> &virial) { _virial = virial; };
};

class VirialMolecular : public Virial
{
    void computeVirial(SimulationBox &simulationBox, PhysicalData &physicalData);
};

class VirialAtomic : public Virial
{
    void computeVirial(SimulationBox &simulationBox, PhysicalData &physicalData){};
};

#endif