#include "resetKinetics.hpp"

#include "constants.hpp"       // for _S_TO_FS_
#include "molecule.hpp"        // for Molecule
#include "physicalData.hpp"    // for PhysicalData, physicalData
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for Vector3D

#include <cmath>     // for sqrt
#include <cstddef>   // for size_t
#include <vector>    // for vector

using namespace resetKinetics;

/**
 * @brief dummy reset function in case no reset is needed
 *
 */
void ResetKinetics::reset(const size_t, physicalData::PhysicalData &, simulationBox::SimulationBox &) const {}

/**
 * @brief reset the momentum of the system
 *
 * @param step
 * @param physicalData
 * @param simBox
 */
void ResetMomentum::reset(const size_t step, physicalData::PhysicalData &physicalData, simulationBox::SimulationBox &simBox) const
{
    if ((step <= _nStepsMomentumReset) || (0 == step % _frequencyMomentumReset))
        ResetKinetics::resetMomentum(physicalData, simBox);
}

/**
 * @brief reset the temperature and the momentum of the system
 *
 * @details reset temperature and momentum if number of steps is smaller than nscale
 *          reset temperature and momentum if number of steps is a multiple of fscale
 *          reset only momentum if number of steps is smaller than nreset
 *          reset only momentum if number of steps is a multiple of freset
 *
 * @param step
 * @param physicalData
 * @param simBox
 */
void ResetTemperature::reset(const size_t                  step,
                             physicalData::PhysicalData   &physicalData,
                             simulationBox::SimulationBox &simBox) const
{
    if ((step <= _nStepsTemperatureReset) || (0 == step % _frequencyTemperatureReset))
    {
        ResetKinetics::resetTemperature(physicalData, simBox);
        ResetKinetics::resetMomentum(physicalData, simBox);
    }
    else if ((step <= _nStepsMomentumReset) || (0 == step % _frequencyMomentumReset))
        ResetKinetics::resetMomentum(physicalData, simBox);
}

/**
 * @brief reset the temperature of the system - hard scaling
 *
 * @details calculate hard scaling factor for target temperature and current temperature and scale all velocities
 *
 * @param physicalData
 * @param simBox
 */
void ResetKinetics::resetTemperature(physicalData::PhysicalData &physicalData, simulationBox::SimulationBox &simBox) const
{
    const auto temperature = physicalData.getTemperature();
    const auto lambda      = ::sqrt(_targetTemperature / temperature);

    for (auto &molecule : simBox.getMolecules())
        molecule.scaleVelocities(lambda);

    physicalData.calculateKineticEnergyAndMomentum(simBox);
}

/**
 * @brief reset the momentum of the system
 *
 * @details subtract momentum correction from all velocities - correction is the total momentum divided by the total mass
 *
 * @param physicalData
 * @param simBox
 */
void ResetKinetics::resetMomentum(physicalData::PhysicalData &physicalData, simulationBox::SimulationBox &simBox) const
{
    const auto momentumVector     = physicalData.getMomentumVector() * constants::_S_TO_FS_;
    const auto momentumCorrection = momentumVector / simBox.getTotalMass();

    for (auto &molecule : simBox.getMolecules())
        molecule.correctVelocities(momentumCorrection);

    physicalData.calculateKineticEnergyAndMomentum(simBox);
    physicalData.calculateTemperature(simBox);
}