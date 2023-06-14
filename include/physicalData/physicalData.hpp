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
    double _volume;
    double _density;

    double _temperature;
    double _pressure;

    Vec3D _momentumVector;
    double _momentum;

    Vec3D _virial;

    Vec3D _kineticEnergyAtomicVector;
    Vec3D _kineticEnergyMolecularVector;
    double _kineticEnergy;

    double _coulombEnergy;
    double _nonCoulombEnergy;

public:
    void calculateKineticEnergyAndMomentum(SimulationBox &);

    void updateAverages(const PhysicalData &);
    void makeAverages(const double);

    std::function<Vec3D()> getKineticEnergyVirialVector = std::bind(&PhysicalData::getKineticEnergyMolecularVector, this);
    void changeKineticVirialToAtomic() { getKineticEnergyVirialVector = std::bind(&PhysicalData::getKineticEnergyAtomicVector, this); }

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

    void setVirial(const Vec3D &virial) { _virial = virial; }
    [[nodiscard]] Vec3D getVirial() const { return _virial; }

    [[nodiscard]] double getKineticEnergy() const { return _kineticEnergy; }

    [[nodiscard]] Vec3D getKineticEnergyAtomicVector() const { return _kineticEnergyAtomicVector; }
    [[nodiscard]] Vec3D getKineticEnergyMolecularVector() const { return _kineticEnergyMolecularVector; }

    void setCoulombEnergy(const double coulombEnergy) { _coulombEnergy = coulombEnergy; }
    [[nodiscard]] double getCoulombEnergy() const { return _coulombEnergy; }

    void setNonCoulombEnergy(const double nonCoulombEnergy) { _nonCoulombEnergy = nonCoulombEnergy; }
    [[nodiscard]] double getNonCoulombEnergy() const { return _nonCoulombEnergy; }
};

#endif // _PHYSICAL_DATA_H_
