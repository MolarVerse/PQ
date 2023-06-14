#ifndef _VIRIAL_H_

#define _VIRIAL_H_

#include "simulationBox.hpp"
#include "physicalData.hpp"
#include "vector3d.hpp"

#include <vector>

class Virial
{
protected:
    Vec3D _virial = Vec3D(0.0, 0.0, 0.0);

public:
    virtual ~Virial() = default;

    virtual void calculateVirial(SimulationBox &simulationBox, PhysicalData &physicalData);

    // standard getter and setters
    [[nodiscard]] Vec3D getVirial() const { return _virial; };
    void setVirial(const Vec3D &virial) { _virial = virial; };
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