#include "virial.hpp"

#include "molecule.hpp"        // for Molecule
#include "physicalData.hpp"    // for PhysicalData, physicalData, simulationBox
#include "simulationBox.hpp"   // for SimulationBox

#include <cstddef>   // for size_t
#include <vector>    // for vector

using namespace simulationBox;
using namespace virial;
using namespace physicalData;

/**
 * @brief calculate virial for general systems
 *
 * @details It calculates the virial for all atoms in the simulation box without any corrections.
 *          It already sets the virial in the physicalData object
 *
 * @param simulationBox
 * @param physicalData
 */
void Virial::calculateVirial(SimulationBox &simulationBox, PhysicalData &physicalData)
{
    _virial = {0.0, 0.0, 0.0};

    for (auto &molecule : simulationBox.getMolecules())
    {
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto forcexyz      = molecule.getAtomForce(i);
            const auto shiftForcexyz = molecule.getAtomShiftForce(i);
            const auto xyz           = molecule.getAtomPosition(i);

            _virial += forcexyz * xyz + shiftForcexyz;

            molecule.setAtomShiftForces(i, {0.0, 0.0, 0.0});
        }
    }

    physicalData.setVirial(_virial);
}

/**
 * @brief calculate virial for molecular systems
 *
 * @details it calls the general virial calculation and then corrects it for
 *          intramolecular interactions. Afterwards it sets the virial in the
 *          physicalData object
 *
 * @param simulationBox
 * @param physicalData
 */
void VirialMolecular::calculateVirial(SimulationBox &simulationBox, PhysicalData &physicalData)
{
    Virial::calculateVirial(simulationBox, physicalData);

    intraMolecularVirialCorrection(simulationBox);

    physicalData.setVirial(_virial);
}

/**
 * @brief calculate intramolecular virial correction
 *
 * @note it directly corrects the virial member variable
 *
 * @param simulationBox
 */
void VirialMolecular::intraMolecularVirialCorrection(SimulationBox &simulationBox)
{
    const auto box = simulationBox.getBoxDimensions();

    for (const auto &molecule : simulationBox.getMolecules())
    {
        const auto   centerOfMass  = molecule.getCenterOfMass();
        const size_t numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto forcexyz = molecule.getAtomForce(i);
            const auto xyz      = molecule.getAtomPosition(i);

            auto dxyz = xyz - centerOfMass;

            dxyz -= box * round(dxyz / box);

            _virial -= forcexyz * dxyz;
        }
    }
}