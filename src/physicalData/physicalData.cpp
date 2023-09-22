#include "physicalData.hpp"

#include "constants.hpp"       // for _FS_TO_S_, _KINETIC_ENERGY_FACTOR_, _TEMPERATURE_FACTOR_
#include "simulationBox.hpp"   // for SimulationBox

#include <algorithm>   // for for_each
#include <cstddef>     // for size_t

using namespace physicalData;

/**
 * @brief Calculates kinetic energy and momentum of the system
 *
 * @param simulationBox
 */
void PhysicalData::calculateKinetics(simulationBox::SimulationBox &simulationBox)
{
    _momentum                     = linearAlgebra::Vec3D();
    _kineticEnergyAtomicVector    = linearAlgebra::Vec3D();
    _kineticEnergyMolecularVector = linearAlgebra::Vec3D();

    auto kineticEnergyAndMomentumOfMolecule = [this](auto &molecule)
    {
        auto momentumSquared = linearAlgebra::Vec3D();

        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            const auto velocities = molecule.getAtomVelocity(i);

            const auto momentum = velocities * molecule.getAtomMass(i);

            _momentum                  += momentum;
            _kineticEnergyAtomicVector += momentum * velocities;
            momentumSquared            += momentum * momentum;
        }

        _kineticEnergyMolecularVector += momentumSquared / molecule.getMolMass();
    };

    std::ranges::for_each(simulationBox.getMolecules(), kineticEnergyAndMomentumOfMolecule);

    _momentum *= constants::_FS_TO_S_;

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
    _kineticEnergy += physicalData.getKineticEnergy();
    _volume        += physicalData.getVolume();
    _density       += physicalData.getDensity();
    _virial        += physicalData.getVirial();
    _pressure      += physicalData.getPressure();

    _qmEnergy += physicalData.getQMEnergy();

    _momentum        += physicalData.getMomentum();
    _angularMomentum += physicalData.getAngularMomentum();

    _noseHooverMomentumEnergy += physicalData.getNoseHooverMomentumEnergy();
    _noseHooverFrictionEnergy += physicalData.getNoseHooverFrictionEnergy();
}

/**
 * @brief calculates the average of all physicalData of last steps
 *
 * @param outputFrequency
 */
void PhysicalData::makeAverages(const double outputFrequency)
{
    _kineticEnergy         /= outputFrequency;
    _coulombEnergy         /= outputFrequency;
    _nonCoulombEnergy      /= outputFrequency;
    _intraCoulombEnergy    /= outputFrequency;
    _intraNonCoulombEnergy /= outputFrequency;

    _bondEnergy     /= outputFrequency;
    _angleEnergy    /= outputFrequency;
    _dihedralEnergy /= outputFrequency;
    _improperEnergy /= outputFrequency;

    _temperature /= outputFrequency;
    _volume      /= outputFrequency;
    _density     /= outputFrequency;
    _virial      /= outputFrequency;
    _pressure    /= outputFrequency;

    _qmEnergy /= outputFrequency;

    _momentum        /= outputFrequency;
    _angularMomentum /= outputFrequency;

    _noseHooverMomentumEnergy /= outputFrequency;
    _noseHooverFrictionEnergy /= outputFrequency;
}

/**
 * @brief clear all physicalData in order to call add functions
 *
 */
void PhysicalData::reset()
{
    _kineticEnergy         = 0.0;
    _coulombEnergy         = 0.0;
    _nonCoulombEnergy      = 0.0;
    _intraCoulombEnergy    = 0.0;
    _intraNonCoulombEnergy = 0.0;

    _bondEnergy     = 0.0;
    _angleEnergy    = 0.0;
    _dihedralEnergy = 0.0;
    _improperEnergy = 0.0;

    _temperature = 0.0;
    _virial      = {0.0, 0.0, 0.0};
    _pressure    = 0.0;
    _volume      = 0.0;
    _density     = 0.0;

    _qmEnergy = 0.0;

    _momentum        = {0.0, 0.0, 0.0};
    _angularMomentum = {0.0, 0.0, 0.0};

    _noseHooverMomentumEnergy = 0.0;
    _noseHooverFrictionEnergy = 0.0;
}

/**
 * @brief calculate temperature
 *
 * @param simulationBox
 */
void PhysicalData::calculateTemperature(simulationBox::SimulationBox &simulationBox)
{
    _temperature = simulationBox.calculateTemperature();
}

/**
 * @brief calculate potential energy
 *
 * @return double
 */
double PhysicalData::getTotalEnergy() const
{
    auto potentialEnergy = 0.0;

    potentialEnergy += _bondEnergy;
    potentialEnergy += _angleEnergy;
    potentialEnergy += _dihedralEnergy;
    potentialEnergy += _improperEnergy;

    potentialEnergy += _coulombEnergy;      // intra + inter
    potentialEnergy += _nonCoulombEnergy;   // intra + inter

    potentialEnergy += _kineticEnergy;

    potentialEnergy += _qmEnergy;

    return potentialEnergy;
}

/**
 * @brief add intra coulomb energy
 *
 * @details This function is used to add intra coulomb energy to the total coulomb energy
 *
 * @param intraCoulombEnergy
 */
void PhysicalData::addIntraCoulombEnergy(const double intraCoulombEnergy)
{
    _intraCoulombEnergy += intraCoulombEnergy;
    _coulombEnergy      += intraCoulombEnergy;
}

/**
 * @brief add intra non coulomb energy
 *
 * @details This function is used to add intra non coulomb energy to the total non coulomb energy
 *
 * @param intraNonCoulombEnergy
 */
void PhysicalData::addIntraNonCoulombEnergy(const double intraNonCoulombEnergy)
{
    _intraNonCoulombEnergy += intraNonCoulombEnergy;
    _nonCoulombEnergy      += intraNonCoulombEnergy;
}

/**
 * @brief change kinetic virial to atomic
 *
 * @details This function is used to change the kinetic virial from molecular to atomic via a function pointer
 *
 */
void PhysicalData::changeKineticVirialToAtomic()
{
    getKineticEnergyVirialVector = std::bind_front(&PhysicalData::getKineticEnergyAtomicVector, this);
}