#include "physicalData.hpp"
#include "mathUtilities.hpp"
#include "constants.hpp"

#include <vector>
#include <cmath>

using namespace std;

void PhysicalData::calculateKineticEnergyAndMomentum(SimulationBox &simulationBox)
{
    vector<double> momentum(3, 0.0);
    vector<double> momentumSquared(3, 0.0);

    _momentumVector = {0.0, 0.0, 0.0};
    _kineticEnergyAtomicVector = {0.0, 0.0, 0.0};
    _kineticEnergyMolecularVector = {0.0, 0.0, 0.0};

    for (auto &molecule : simulationBox._molecules)
    {
        momentumSquared[0] = 0.0;
        momentumSquared[1] = 0.0;
        momentumSquared[2] = 0.0;

        for (int i = 0; i < molecule.getNumberOfAtoms(); i++)
        {
            auto velocities = molecule.getAtomVelocities(i);
            auto mass = molecule.getMass(i);

            momentum[0] = velocities[0] * mass;
            momentum[1] = velocities[1] * mass;
            momentum[2] = velocities[2] * mass;

            _momentumVector[0] += momentum[0];
            _momentumVector[1] += momentum[1];
            _momentumVector[2] += momentum[2];

            _kineticEnergyAtomicVector[0] += momentum[0] * velocities[0];
            _kineticEnergyAtomicVector[1] += momentum[1] * velocities[1];
            _kineticEnergyAtomicVector[2] += momentum[2] * velocities[2];

            momentumSquared[0] += momentum[0] * momentum[0];
            momentumSquared[1] += momentum[1] * momentum[1];
            momentumSquared[2] += momentum[2] * momentum[2];
        }

        _kineticEnergyMolecularVector[0] += momentumSquared[0] / molecule.getMolMass();
        _kineticEnergyMolecularVector[1] += momentumSquared[1] / molecule.getMolMass();
        _kineticEnergyMolecularVector[2] += momentumSquared[2] / molecule.getMolMass();
    }

    _momentumVector[0] *= _FS_TO_S_;
    _momentumVector[1] *= _FS_TO_S_;
    _momentumVector[2] *= _FS_TO_S_;
    _momentum = sqrt(_momentumVector[0] * _momentumVector[0] + _momentumVector[1] * _momentumVector[1] + _momentumVector[2] * _momentumVector[2]);

    _kineticEnergyAtomicVector[0] *= _KINETIC_ENERGY_FACTOR_;
    _kineticEnergyAtomicVector[1] *= _KINETIC_ENERGY_FACTOR_;
    _kineticEnergyAtomicVector[2] *= _KINETIC_ENERGY_FACTOR_;
    _kineticEnergyMolecularVector[0] *= _KINETIC_ENERGY_FACTOR_;
    _kineticEnergyMolecularVector[1] *= _KINETIC_ENERGY_FACTOR_;
    _kineticEnergyMolecularVector[2] *= _KINETIC_ENERGY_FACTOR_;

    _kineticEnergy = _kineticEnergyAtomicVector[0] + _kineticEnergyAtomicVector[1] + _kineticEnergyAtomicVector[2];
}

void PhysicalData::updateAverages(const PhysicalData &physicalData)
{
    _coulombEnergy += physicalData.getCoulombEnergy();
    _nonCoulombEnergy += physicalData.getNonCoulombEnergy();
    _temperature += physicalData.getTemperature();
    _momentum += physicalData.getMomentum();
    _kineticEnergy += physicalData.getKineticEnergy();
    _volume += physicalData.getVolume();
    _density += physicalData.getDensity();

    _virial[0] += physicalData.getVirial()[0];
    _virial[1] += physicalData.getVirial()[1];
    _virial[2] += physicalData.getVirial()[2];

    _pressure += physicalData.getPressure();
}

void PhysicalData::resetData()
{
    _coulombEnergy = 0.0;
    _nonCoulombEnergy = 0.0;
    _temperature = 0.0;
    _momentum = 0.0;
    _kineticEnergy = 0.0;
}