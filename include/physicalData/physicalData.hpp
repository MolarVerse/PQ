#ifndef _PHYSICAL_DATA_H_

#define _PHYSICAL_DATA_H_

#include <vector>
#include <functional>

#include "simulationBox.hpp"

/**
 * @class PhysicalData
 *
 * @brief PhysicalData is a class for output data storage
 *
 */
class PhysicalData
{
private:
    double _volume = 0.0;
    double _density = 0.0;

    double _temperature = 0.0;
    double _pressure = 0.0;

    std::vector<double> _momentumVector = {0.0, 0.0, 0.0};
    double _momentum = 0.0;

    std::vector<double> _virial = {0.0, 0.0, 0.0};

    std::vector<double> _kineticEnergyAtomicVector = {0.0, 0.0, 0.0};
    std::vector<double> _kineticEnergyMolecularVector = {0.0, 0.0, 0.0};
    double _kineticEnergy = 0.0;

    double _coulombEnergy = 0.0;
    double _nonCoulombEnergy = 0.0;

public:
    void calculateKineticEnergyAndMomentum(SimulationBox &);

    void updateAverages(const PhysicalData &);

    void resetData(); // not used at the moment

    std::function<std::vector<double>()> getKineticEnergyVirialVector = std::bind(&PhysicalData::getKineticEnergyMolecularVector, this);
    void changeKineticVirialToAtomic()
    {
        getKineticEnergyVirialVector = std::bind(&PhysicalData::getKineticEnergyAtomicVector, this);
    }

    // standard getter and setters
    void setVolume(const double volume) { _volume = volume; }
    [[nodiscard]] double getVolume() const { return _volume; }

    void setDensity(const double density) { _density = density; }
    [[nodiscard]] double getDensity() const { return _density; }

    void setTemperature(const double temperature) { _temperature = temperature; }
    [[nodiscard]] double getTemperature() const { return _temperature; }

    void setPressure(const double pressure) { _pressure = pressure; }
    [[nodiscard]] double getPressure() const { return _pressure; }

    [[nodiscard]] double getMomentum() const { return _momentum; }

    void setVirial(const std::vector<double> &virial) { _virial = virial; }
    std::vector<double> getVirial() const { return _virial; }

    [[nodiscard]] double getKineticEnergy() const { return _kineticEnergy; }

    std::vector<double> getKineticEnergyAtomicVector() const { return _kineticEnergyAtomicVector; }
    std::vector<double> getKineticEnergyMolecularVector() const { return _kineticEnergyMolecularVector; }

    void setCoulombEnergy(const double coulombEnergy) { _coulombEnergy = coulombEnergy; }
    [[nodiscard]] double getCoulombEnergy() const { return _coulombEnergy; }

    void setNonCoulombEnergy(const double nonCoulombEnergy) { _nonCoulombEnergy = nonCoulombEnergy; }
    [[nodiscard]] double getNonCoulombEnergy() const { return _nonCoulombEnergy; }
};

#endif // _PHYSICAL_DATA_H_
