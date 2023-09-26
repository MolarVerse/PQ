#include "mmmdEngine.hpp"

#include "celllist.hpp"          // for CellList
#include "constraints.hpp"       // for Constraints
#include "engineOutput.hpp"      // for engine
#include "forceFieldClass.hpp"   // for ForceField
#include "integrator.hpp"        // for Integrator
#include "intraNonBonded.hpp"    // for IntraNonBonded
#include "manostat.hpp"          // for Manostat
#include "physicalData.hpp"      // for PhysicalData
#include "potential.hpp"         // for Potential
#include "resetKinetics.hpp"     // for ResetKinetics
#include "thermostat.hpp"        // for Thermostat
#include "virial.hpp"            // for Virial

#include <memory>   // for unique_ptr

using namespace engine;

/**
 * @brief Takes one step in the simulation.
 *
 * @details The step is taken in the following order:
 * - First step of the integrator
 * - Apply SHAKE
 * - Update cell list
 * - Calculate forces
 * - Calculate intra non bonded forces
 * - Calculate virial
 * - Calculate constraint bond references
 * - calculate intra molecular virial correction
 * - Apply thermostat on forces
 * - Second step of the integrator
 * - Apply RATTLE
 * - Apply thermostat
 * - Calculate kinetic energy and momentum
 * - Apply manostat
 * - Reset temperature and momentum
 *
 */
void MMMDEngine::takeStep()
{
    _thermostat->applyThermostatHalfStep(_simulationBox, _physicalData);

    _integrator->firstStep(_simulationBox);

    _constraints.applyShake(_simulationBox);

    _cellList.updateCellList(_simulationBox);

    _potential->calculateForces(_simulationBox, _physicalData, _cellList);

    _intraNonBonded.calculate(_simulationBox, _physicalData);

    _virial->calculateVirial(_simulationBox, _physicalData);

    _forceField.calculateBondedInteractions(_simulationBox, _physicalData);

    _constraints.calculateConstraintBondRefs(_simulationBox);

    _virial->intraMolecularVirialCorrection(_simulationBox, _physicalData);

    _thermostat->applyThermostatOnForces(_simulationBox);

    _integrator->secondStep(_simulationBox);

    _constraints.applyRattle();

    _thermostat->applyThermostat(_simulationBox, _physicalData);

    _physicalData.calculateKinetics(_simulationBox);

    _manostat->applyManostat(_simulationBox, _physicalData);

    _resetKinetics.reset(_step, _physicalData, _simulationBox);
}