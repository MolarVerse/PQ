#ifndef _MANOSTAT_H_

#define _MANOSTAT_H_

#include "virial.hpp"
#include "physicalData.hpp"

class Manostat
{
protected:
    double _pressure;

public:
    void calculatePressure(Virial &Virial, PhysicalData &physicalData);
};

#endif