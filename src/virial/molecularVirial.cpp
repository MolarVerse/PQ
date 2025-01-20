/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "molecularVirial.hpp"

#include "debug.hpp"
#include "linearAlgebra.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "triclinicBox.hpp"

using namespace virial;
using namespace simulationBox;
using namespace physicalData;
using namespace linearAlgebra;

/**
 * @brief Construct a new Virial Molecular:: Virial Molecular object
 *
 */
MolecularVirial::MolecularVirial() : Virial() { _virialType = "molecular"; }

/**
 * @brief clones the molecular virial object
 *
 * @return std::shared_ptr<Virial>
 */
std::shared_ptr<Virial> MolecularVirial::clone() const
{
    return std::make_shared<MolecularVirial>(*this);
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
void MolecularVirial::calculateVirial(
    SimulationBox &simulationBox,
    PhysicalData  &physicalData
)
{
    Virial::calculateVirial(simulationBox, physicalData);

    physicalData.setVirial(_virial);
}

/**
 * @brief calculate intramolecular virial correction
 *
 * @note it directly corrects the virial member variable
 *
 * @param simBox
 * @param data
 */
void MolecularVirial::intraMolecularVirialCorrection(
    SimulationBox &simBox,
    PhysicalData  &data
)
{
    startTimingsSection("IntraMolecular Correction");

    __DEBUG_ENTER_FUNCTION__("IntraMolecular Correction");

    Real virial[9] = {0.0};

    const auto        nMolecules     = simBox.getNumberOfMolecules();
    const auto *const comMolecules   = simBox.getComMoleculesPtr();
    const auto *const atomsPerMolPtr = simBox.getAtomsPerMoleculePtr();
    const auto *const molOffsetPtr   = simBox.getMoleculeOffsetsPtr();
    const auto *const forcesPtr      = simBox.getForcesPtr();
    const auto *const posPtr         = simBox.getPosPtr();
    const auto *const boxParams      = simBox.getBox().getBoxParamsPtr();
    const auto        isOrthorhombic = simBox.getBox().isOrthoRhombic();

#ifdef __PQ_GPU__
    // clang-format off
    #pragma omp target teams distribute parallel for \
                is_device_ptr(comMolecules, atomsPerMolPtr, \
                              molOffsetPtr, forcesPtr, posPtr, \
                              boxParams)                      \
                map(virial)
#else
    #pragma omp parallel for
    // clang-format on
#endif
    for (size_t i = 0; i < nMolecules; ++i)
    {
        const auto nAtoms    = atomsPerMolPtr[i];
        const auto comX      = comMolecules[i * 3];
        const auto comY      = comMolecules[i * 3 + 1];
        const auto comZ      = comMolecules[i * 3 + 2];
        const auto molOffset = molOffsetPtr[i];

        for (size_t j = 0; j < nAtoms; ++j)
        {
            const auto atomIndex = molOffset + j;
            const auto posX      = posPtr[atomIndex * 3];
            const auto posY      = posPtr[atomIndex * 3 + 1];
            const auto posZ      = posPtr[atomIndex * 3 + 2];
            const auto forceX    = forcesPtr[atomIndex * 3];
            const auto forceY    = forcesPtr[atomIndex * 3 + 1];
            const auto forceZ    = forcesPtr[atomIndex * 3 + 2];

            auto dx = posX - comX;
            auto dy = posY - comY;
            auto dz = posZ - comZ;

            if (isOrthorhombic)
                imageOrthoRhombic(boxParams, dx, dy, dz);
            else
                imageTriclinic(boxParams, dx, dy, dz);

            Real help[9] = {0.0};
            tensorProduct(help, dx, dy, dz, forceX, forceY, forceZ);

            for (size_t k = 0; k < 9; ++k)
                atomicSubtract(&virial[k], help[k]);
        }
    }

    _virial = tensor3D{virial};
    data.addVirial(_virial);

    __DEBUG_VIRIAL__(data.getVirial());
    __DEBUG_EXIT_FUNCTION__("IntraMolecular Correction");

    stopTimingsSection("IntraMolecular Correction");
}