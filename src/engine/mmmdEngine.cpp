#include "mmmdEngine.hpp"

#include "constants.hpp"         // for _FS_TO_PS_
#include "logOutput.hpp"         // for LogOutput
#include "output.hpp"            // for Output
#include "progressbar.hpp"       // for progressbar
#include "stdoutOutput.hpp"      // for StdoutOutput
#include "timingsSettings.hpp"   // for TimingsSettings

using namespace engine;

/**
 * @brief Takes one step in the simulation.
 *
 * @details The step is taken in the following order:
 *  1.  First step of the integrator
 *  2.  Apply SHAKE
 *  3.  Update cell list
 *  4.1 Calculate forces
 *  4.2 Calculate intra non bonded forces
 *  5.  Calculate virial
 *  6.  Calculate constraint bond references
 *  7.  Second step of the integrator
 *  8.  calculate intra molecular virial correction
 *  9.  Apply RATTLE
 * 10.  Apply thermostat
 * 11.  Calculate kinetic energy and momentum
 * 12.  Apply manostat
 * 13.  Reset temperature and momentum
 *
 */
void MMMDEngine::takeStep()
{
    _integrator->firstStep(_simulationBox);

    _constraints.applyShake(_simulationBox);

    _cellList.updateCellList(_simulationBox);

    _potential->calculateForces(_simulationBox, _physicalData, _cellList);

    _intraNonBonded.calculate(_simulationBox, _physicalData);

    _virial->calculateVirial(_simulationBox, _physicalData);

    _forceField.calculateBondedInteractions(_simulationBox, _physicalData);

    _constraints.calculateConstraintBondRefs(_simulationBox);

    _virial->intraMolecularVirialCorrection(_simulationBox, _physicalData);

    _integrator->secondStep(_simulationBox);

    _constraints.applyRattle();

    _thermostat->applyThermostat(_simulationBox, _physicalData);

    _physicalData.calculateKineticEnergyAndMomentum(_simulationBox);

    _manostat->applyManostat(_simulationBox, _physicalData);

    _resetKinetics->reset(_step, _physicalData, _simulationBox);
}