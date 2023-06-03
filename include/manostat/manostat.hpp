#ifndef _MANOSTAT_H_

#define _MANOSTAT_H_

#include "virial.hpp"
#include "physicalData.hpp"

#include <vector>

class Manostat
{
protected:
    Vec3D _pressureVector = {0.0, 0.0, 0.0};
    double _pressure;

public:
    void calculatePressure(PhysicalData &physicalData);
};

#endif