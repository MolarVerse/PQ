#include "resetKinetics.hpp"

#include "constants.hpp"

using namespace std;
using namespace resetKinetics;
using namespace physicalData;

/**
 * @brief dummy reset function in case no reset is needed
 *
 */
void ResetKinetics::reset(const size_t, PhysicalData &, simulationBox::SimulationBox &) const {}

/**
 * @brief reset the momentum of the system
 *
 * @note important to recalculate the new temperature after the reset
 *
 * @param step
 * @param physicalData
 * @param simBox
 */
void ResetMomentum::reset(const size_t step, PhysicalData &physicalData, simulationBox::SimulationBox &simBox) const
{
    if ((step <= _nStepsMomentumReset) || (step % _frequencyMomentumReset == 0))
    {
        ResetKinetics::resetMomentum(physicalData, simBox);
        physicalData.calculateTemperature(simBox);
    }
}

/**
 * @brief reset the temperature and the momentum of the system
 *
 * @note important to recalculate the momentum after the temperature reset
 *       and the temperature after the momentum reset
 *
 * @param step
 * @param physicalData
 * @param simBox
 */
void ResetTemperature::reset(const size_t step, PhysicalData &physicalData, simulationBox::SimulationBox &simBox) const
{
    if ((step <= _nStepsTemperatureReset) || (step % _frequencyTemperatureReset == 0))
    {
        ResetKinetics::resetTemperature(physicalData, simBox);
        physicalData.calculateKineticEnergyAndMomentum(simBox);
        ResetKinetics::resetMomentum(physicalData, simBox);
        physicalData.calculateTemperature(simBox);
    }
    else if ((step <= _nStepsMomentumReset) || (step % _frequencyMomentumReset == 0))
    {
        ResetKinetics::resetMomentum(physicalData, simBox);
        physicalData.calculateTemperature(simBox);
    }
}

/**
 * @brief reset the temperature of the system - hard scaling
 *
 * @param physicalData
 * @param simBox
 */
void ResetKinetics::resetTemperature(const PhysicalData &physicalData, simulationBox::SimulationBox &simBox) const
{
    const auto temperature = physicalData.getTemperature();
    const auto lambda      = ::sqrt(_targetTemperature / temperature);

    for (auto &molecule : simBox.getMolecules())
        molecule.scaleVelocities(lambda);
}

/**
 * @brief reset the momentum of the system
 *
 * @param physicalData
 * @param simBox
 */
void ResetKinetics::resetMomentum(PhysicalData &physicalData, simulationBox::SimulationBox &simBox) const
{
    const auto momentumVector     = physicalData.getMomentumVector() * constants::_S_TO_FS_;
    const auto momentumCorrection = momentumVector / simBox.getTotalMass();

    for (auto &molecule : simBox.getMolecules())
        molecule.correctVelocities(momentumCorrection);

    physicalData.calculateKineticEnergyAndMomentum(simBox);
}