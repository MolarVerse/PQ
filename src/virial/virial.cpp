#include "virial.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using namespace simulationBox;
using namespace virial;
using namespace physicalData;

/**
 * @brief calculate virial for general systems
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
 * @note
 *  This function is called by VirialMolecular::calculateVirial
 *  it includes also intramolecular virial correction
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