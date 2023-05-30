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

    virtual void calculateVirial(SimulationBox &simulationBox, PhysicalData &physicalData);

    // standard getter and setters
    std::vector<double> getVirial() const { return _virial; };
    void setVirial(const std::vector<double> &virial) { _virial = virial; };
};

class VirialMolecular : public Virial
{
    void calculateVirial(SimulationBox &simulationBox, PhysicalData &physicalData) override;
    void intraMolecularVirialCorrection(SimulationBox &simulationBox);
};

class VirialAtomic : public Virial
{
};

#endif