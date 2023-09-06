#include "physicalData.hpp"

#include "constants.hpp"
#include "simulationBox.hpp"   // for SimulationBox

#include <algorithm>
#include <cstddef>   // for size_t

using namespace physicalData;

/**
 * @brief Calculates kinetic energy and momentum of the system
 *
 * @param simulationBox
 */
void PhysicalData::calculateKineticEnergyAndMomentum(simulationBox::SimulationBox &simulationBox)
{
    _momentumVector               = linearAlgebra::Vec3D();
    _kineticEnergyAtomicVector    = linearAlgebra::Vec3D();
    _kineticEnergyMolecularVector = linearAlgebra::Vec3D();

    auto kineticEnergyAndMomentumOfMolecule = [this](auto &molecule)
    {
        auto momentumSquared = linearAlgebra::Vec3D();

        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            const auto velocities = molecule.getAtomVelocity(i);

            const auto momentum = velocities * molecule.getAtomMass(i);

            _momentumVector            += momentum;
            _kineticEnergyAtomicVector += momentum * velocities;
            momentumSquared            += momentum * momentum;
        }

        _kineticEnergyMolecularVector += momentumSquared / molecule.getMolMass();
    };

    std::ranges::for_each(simulationBox.getMolecules(), kineticEnergyAndMomentumOfMolecule);

    _momentumVector *= constants::_FS_TO_S_;
    _momentum        = norm(_momentumVector);

    _kineticEnergyAtomicVector    *= constants::_KINETIC_ENERGY_FACTOR_;
    _kineticEnergyMolecularVector *= constants::_KINETIC_ENERGY_FACTOR_;
    _kineticEnergy                 = sum(_kineticEnergyAtomicVector);
}

/**
 * @brief calculates the sum of all physicalData of last steps
 *
 * @param physicalData
 */
void PhysicalData::updateAverages(const PhysicalData &physicalData)
{
    _coulombEnergy         += physicalData.getCoulombEnergy();
    _nonCoulombEnergy      += physicalData.getNonCoulombEnergy();
    _intraCoulombEnergy    += physicalData.getIntraCoulombEnergy();
    _intraNonCoulombEnergy += physicalData.getIntraNonCoulombEnergy();

    _bondEnergy     += physicalData.getBondEnergy();
    _angleEnergy    += physicalData.getAngleEnergy();
    _dihedralEnergy += physicalData.getDihedralEnergy();
    _improperEnergy += physicalData.getImproperEnergy();

    _temperature   += physicalData.getTemperature();
    _momentum      += physicalData.getMomentum();
    _kineticEnergy += physicalData.getKineticEnergy();
    _volume        += physicalData.getVolume();
    _density       += physicalData.getDensity();
    _virial        += physicalData.getVirial();
    _pressure      += physicalData.getPressure();
}

/**
 * @brief calculates the average of all physicalData of last steps
 *
 * @param outputFrequency
 */
void PhysicalData::makeAverages(const double outputFrequency)
{
    _coulombEnergy         /= outputFrequency;
    _nonCoulombEnergy      /= outputFrequency;
    _intraCoulombEnergy    /= outputFrequency;
    _intraNonCoulombEnergy /= outputFrequency;

    _bondEnergy     /= outputFrequency;
    _angleEnergy    /= outputFrequency;
    _dihedralEnergy /= outputFrequency;
    _improperEnergy /= outputFrequency;

    _temperature   /= outputFrequency;
    _momentum      /= outputFrequency;
    _kineticEnergy /= outputFrequency;
    _volume        /= outputFrequency;
    _density       /= outputFrequency;
    _virial        /= outputFrequency;
    _pressure      /= outputFrequency;
}

/**
 * @brief clear all physicalData in order to call add functions
 *
 */
void PhysicalData::clearData()
{
    _coulombEnergy         = 0.0;
    _nonCoulombEnergy      = 0.0;
    _intraCoulombEnergy    = 0.0;
    _intraNonCoulombEnergy = 0.0;

    _bondEnergy     = 0.0;
    _angleEnergy    = 0.0;
    _dihedralEnergy = 0.0;
    _improperEnergy = 0.0;

    _temperature   = 0.0;
    _momentum      = 0.0;
    _kineticEnergy = 0.0;
    _virial        = {0.0, 0.0, 0.0};
    _pressure      = 0.0;
}

/**
 * @brief calculate temperature
 *
 * @param simulationBox
 */
void PhysicalData::calculateTemperature(simulationBox::SimulationBox &simulationBox)
{
    _temperature = 0.0;

    auto temperatureOfMolecule = [this](auto &molecule)
    {
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            const auto velocities = molecule.getAtomVelocity(i);
            const auto mass       = molecule.getAtomMass(i);

            _temperature += mass * normSquared(velocities);
        }
    };

    std::ranges::for_each(simulationBox.getMolecules(), temperatureOfMolecule);

    _temperature *= constants::_TEMPERATURE_FACTOR_ / double(simulationBox.getDegreesOfFreedom());
}

/**
 * @brief calculate potential energy
 *
 * @return double
 */
double PhysicalData::getPotentialEnergy() const
{
    auto potentialEnergy = 0.0;

    potentialEnergy += _bondEnergy;
    potentialEnergy += _angleEnergy;
    potentialEnergy += _dihedralEnergy;
    potentialEnergy += _improperEnergy;

    potentialEnergy += _coulombEnergy;      // intra + inter
    potentialEnergy += _nonCoulombEnergy;   // intra + inter

    return potentialEnergy;
}

/**
 * @brief calculate intra energy
 *
 * @return double
 */
double PhysicalData::getIntraEnergy() const
{
    auto intraEnergy = 0.0;

    intraEnergy += _intraCoulombEnergy;
    intraEnergy += _intraNonCoulombEnergy;

    return intraEnergy;
}