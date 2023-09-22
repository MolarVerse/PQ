#include "resetKinetics.hpp"

#include "constants.hpp"            // for _S_TO_FS_
#include "physicalData.hpp"         // for PhysicalData
#include "simulationBox.hpp"        // for SimulationBox
#include "staticMatrix3x3.hpp"      // for StaticMatrix3x3
#include "thermostatSettings.hpp"   // for ThermostatSettings
#include "vector3d.hpp"             // for Vector3D

#include <algorithm>    // for __for_each_fn, for_each
#include <cmath>        // for sqrt
#include <cstddef>      // for size_t
#include <functional>   // for identity

using namespace resetKinetics;
using linearAlgebra::diagonalMatrix;
using linearAlgebra::inverse;
using linearAlgebra::StaticMatrix3x3;
using linearAlgebra::trace;
using linearAlgebra::vectorProduct;

/**
 * @brief checks to reset angular momentum
 *
 */
void ResetKinetics::reset(const size_t                  step,
                          physicalData::PhysicalData   &physicalData,
                          simulationBox::SimulationBox &simulationBox) const
{
    if ((step <= _nStepsAngularReset) || (0 == step % _frequencyAngularReset))
        ResetKinetics::resetAngularMomentum(physicalData, simulationBox);
}

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

    ResetKinetics::reset(step, physicalData, simBox);
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

    ResetKinetics::reset(step, physicalData, simBox);
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
    const auto targetTemperature = settings::ThermostatSettings::getTargetTemperature();
    const auto temperature       = physicalData.getTemperature();
    const auto lambda            = ::sqrt(targetTemperature / temperature);

    std::ranges::for_each(simBox.getAtoms(), [lambda](auto &atom) { atom->scaleVelocity(lambda); });

    physicalData.calculateKinetics(simBox);
    physicalData.calculateTemperature(simBox);
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

    std::ranges::for_each(simBox.getAtoms(), [momentumCorrection](auto &atom) { atom->addVelocity(-momentumCorrection); });

    physicalData.calculateKinetics(simBox);
    physicalData.calculateTemperature(simBox);
}

void ResetKinetics::resetAngularMomentum(physicalData::PhysicalData &physicalData, simulationBox::SimulationBox &simBox) const
{
    simBox.calculateTotalMass();
    simBox.calculateCenterOfMass();
    simBox.calculateMomentum();
    simBox.calculateAngularMomentum();

    const auto centerOfMass = simBox.getCenterOfMass();

    StaticMatrix3x3 helperMatrix{0.0};

    auto addInertiaTensorOfAtom = [&helperMatrix, &centerOfMass](const auto &atom)
    {
        auto relativePosition  = atom->getPosition() - centerOfMass;
        helperMatrix          += vectorProduct(relativePosition, relativePosition) * atom->getMass();
    };

    std::ranges::for_each(simBox.getAtoms(), addInertiaTensorOfAtom);

    const StaticMatrix3x3 inertiaTensor        = -helperMatrix + diagonalMatrix(trace(helperMatrix));
    const StaticMatrix3x3 inverseInertiaTensor = inverse(inertiaTensor);

    const auto angularVelocity = inverseInertiaTensor * simBox.getAngularMomentum();

    auto correctVelocities = [&angularVelocity, &centerOfMass](auto &atom)
    {
        auto relativePosition = atom->getPosition() - centerOfMass;
        atom->addVelocity(-cross(angularVelocity, relativePosition));
    };

    std::ranges::for_each(simBox.getAtoms(), correctVelocities);

    physicalData.calculateKinetics(simBox);
    physicalData.calculateTemperature(simBox);
    simBox.calculateMomentum();
    simBox.calculateAngularMomentum();
    physicalData.setAngularMomentum(simBox.getAngularMomentum());
}