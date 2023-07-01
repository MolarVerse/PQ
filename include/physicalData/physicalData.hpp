#ifndef _PHYSICAL_DATA_H_

#define _PHYSICAL_DATA_H_

#include "simulationBox.hpp"

#include <functional>
#include <vector>

namespace physicalData
{
    class PhysicalData;
}

/**
 * @class PhysicalData
 *
 * @brief PhysicalData is a class for output data storage
 *
 */
class physicalData::PhysicalData
{
  private:
    double _volume;
    double _density;
    double _temperature;
    double _pressure;
    double _momentum;
    double _kineticEnergy;
    double _coulombEnergy;
    double _nonCoulombEnergy;

    vector3d::Vec3D _virial;
    vector3d::Vec3D _momentumVector;
    vector3d::Vec3D _kineticEnergyAtomicVector;
    vector3d::Vec3D _kineticEnergyMolecularVector;

  public:
    void calculateTemperature(simulationBox::SimulationBox &);

    void calculateKineticEnergyAndMomentum(simulationBox::SimulationBox &);

    void updateAverages(const PhysicalData &);
    void makeAverages(const double);

    std::function<vector3d::Vec3D()> getKineticEnergyVirialVector =
        std::bind(&PhysicalData::getKineticEnergyMolecularVector, this);

    void changeKineticVirialToAtomic()
    {
        getKineticEnergyVirialVector = std::bind(&PhysicalData::getKineticEnergyAtomicVector, this);
    }

    /********************
     * standard setters *
     ********************/

    void setVolume(const double volume) { _volume = volume; }
    void setDensity(const double density) { _density = density; }
    void setTemperature(const double temperature) { _temperature = temperature; }
    void setPressure(const double pressure) { _pressure = pressure; }
    void setVirial(const vector3d::Vec3D &virial) { _virial = virial; }
    void setCoulombEnergy(const double coulombEnergy) { _coulombEnergy = coulombEnergy; }
    void setNonCoulombEnergy(const double nonCoulombEnergy) { _nonCoulombEnergy = nonCoulombEnergy; }
    void setMomentum(const double momentum) { _momentum = momentum; }
    void setKineticEnergy(const double kineticEnergy) { _kineticEnergy = kineticEnergy; }
    void setKineticEnergyAtomicVector(const vector3d::Vec3D &vec) { _kineticEnergyAtomicVector = vec; }
    void setKineticEnergyMolecularVector(const vector3d::Vec3D &vec) { _kineticEnergyMolecularVector = vec; }

    /********************
     * standard getters *
     ********************/

    double          getVolume() const { return _volume; }
    double          getDensity() const { return _density; }
    double          getTemperature() const { return _temperature; }
    double          getPressure() const { return _pressure; }
    double          getMomentum() const { return _momentum; }
    double          getNonCoulombEnergy() const { return _nonCoulombEnergy; }
    double          getCoulombEnergy() const { return _coulombEnergy; }
    double          getKineticEnergy() const { return _kineticEnergy; }
    vector3d::Vec3D getKineticEnergyAtomicVector() const { return _kineticEnergyAtomicVector; }
    vector3d::Vec3D getKineticEnergyMolecularVector() const { return _kineticEnergyMolecularVector; }
    vector3d::Vec3D getVirial() const { return _virial; }
    vector3d::Vec3D getMomentumVector() const { return _momentumVector; }
};

#endif   // _PHYSICAL_DATA_H_
