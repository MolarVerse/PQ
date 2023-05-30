#ifndef _MANOSTAT_H_

#define _MANOSTAT_H_

#include "virial.hpp"
#include "physicalData.hpp"

#include <vector>

class Manostat
{
protected:
    std::vector<double> _pressureVector = {0.0, 0.0, 0.0};
    double _pressure;

public:
    void calculatePressure(const Virial &Virial, PhysicalData &physicalData);
};

#endif