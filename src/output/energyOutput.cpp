#include "energyOutput.hpp"

#include <iomanip>

using namespace std;
using namespace physicalData;

void EnergyOutput::write(const size_t step, const PhysicalData &data)
{
    _fp << right;
    _fp << setw(10);
    _fp << step;

    _fp << fixed;
    _fp << setprecision(12);

    _fp << right;
    _fp << setw(20);
    _fp << data.getTemperature();

    _fp << right;
    _fp << setw(20);
    _fp << data.getPressure();

    _fp << right;
    _fp << setw(20);
    _fp << 0.0;

    _fp << right;
    _fp << setw(20);
    _fp << data.getKineticEnergy();

    _fp << right;
    _fp << setw(20);
    _fp << 0.0;

    _fp << right;
    _fp << setw(20);
    _fp << data.getCoulombEnergy();

    _fp << right;
    _fp << setw(20);
    _fp << data.getNonCoulombEnergy();

    _fp << right;
    _fp << setw(20);
    _fp << 0.0;

    _fp << right;
    _fp << setw(20);
    _fp << 0.0;

    _fp << scientific;
    _fp << right;
    _fp << setw(20);
    _fp << data.getMomentum();

    _fp << fixed;
    _fp << right;
    _fp << setw(20);
    _fp << 0.0;
    _fp << endl;
}