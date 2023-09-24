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