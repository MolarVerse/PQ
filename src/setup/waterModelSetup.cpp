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

#include "waterModelSetup.hpp"

#include <format>
#include <unordered_set>

#include "engine.hpp"               // for Engine
#include "exceptions.hpp"           // for customException
#include "fileSettings.hpp"         // for FileSettings
#include "mdEngine.hpp"             // for MDEngine
#include "molecule.hpp"             // for Molecule
#include "waterModelSettings.hpp"   // for WaterModelSettings

using namespace setup;
using namespace settings;
using namespace engine;
using namespace customException;

/**
 * @brief wrapper for water model setup
 *
 * @details constructs a water model setup object and calls the setup function
 *
 * @param engine
 */
void setup::setupWaterModel(Engine &engine)
{
    engine.getStdoutOutput().writeSetup("Water model");
    engine.getLogOutput().writeSetup("Water model");

    WaterModelSetup waterModelSetup(dynamic_cast<MDEngine &>(engine));
    waterModelSetup.setup();
}

/**
 * @brief Construct a new Water Model Setup object
 *
 * @param engine
 */
WaterModelSetup::WaterModelSetup(MDEngine &engine) : _engine(engine) {}

/**
 * @brief setup water model
 *
 * @throw UserInputException if water model is requested but no water type is
 * specified
 * @throw MolDescriptorException if water molecule doesn't have exactly 3 atoms
 * in the required order (O, H, H)
 */
void WaterModelSetup::setup()
{
    const auto waterType = _engine.getSimulationBox().getWaterType();

    if (!waterType.has_value())
        throw(UserInputException(
            "Use of water model has been requested in the input file, but "
            "no water type is specified in the moldescriptor file."
        ));

    const auto water =
        _engine.getSimulationBox().findMoleculeType(waterType.value());

    // water atoms have to be in this order for calculation
    if (water.getAtomNames() != pq::strings{"O", "H", "H"})
        throw(MolDescriptorException(
            "Water molecule type must have exactly 3 atoms in the following "
            "order: O (oxygen), H (hydrogen), H (hydrogen)."
        ));

    if (Settings::isQMOnlyJobtype())
        throw(UserInputException(
            "Water models are not supported for QM-only job types."
        ));

    checkTopologyFile();

    _engine.getLogOutput().writeSetupInfo(
        std::format(
            "Intramolecular water model: {}",
            string(WaterModelSettings::getWaterIntraModel())
        )
    );

    _engine.getLogOutput().writeSetupInfo(
        std::format(
            "Intermolecular water model: {}",
            string(WaterModelSettings::getWaterInterModel())
        )
    );

    _engine.getLogOutput().writeEmptyLine();
}

void WaterModelSetup::checkTopologyFile()
{
    std::unordered_set<const pq::Molecule *> waterMolecules;
    for (const auto &waterMol :
         _engine.getSimulationBox().getWaterTypeMolecules())
        waterMolecules.insert(&waterMol);

    size_t bondIndex = 1;

    for (const auto &bond : _engine.getForceField().getBonds())
    {
        const auto *mol1 = bond.getMolecule1();
        const auto *mol2 = bond.getMolecule2();

        const bool involvesWater =
            (mol1 && waterMolecules.find(mol1) != waterMolecules.end()) ||
            (mol2 && waterMolecules.find(mol2) != waterMolecules.end());

        if (involvesWater)
            throw(UserInputException(
                std::format(
                    "A water type molecule is included in the bond list of the "
                    "topology file \"{}\" at entry number {}. Requesting the "
                    "use of the \"{}\" intramolecular water type model expects "
                    "the molecules of this moltype not to appear in the "
                    "topology file.",
                    FileSettings::getTopologyFileName(),
                    bondIndex,
                    string(WaterModelSettings::getWaterIntraModel())
                )
            ));

        ++bondIndex;
    }

    size_t angleIndex = 1;

    for (const auto &angle : _engine.getForceField().getAngles())
    {
        const auto  molecules = angle.getMolecules();
        const auto *mol1      = molecules[0];
        const auto *mol2      = molecules[1];
        const auto *mol3      = molecules[2];

        const bool involvesWater =
            (mol1 && waterMolecules.find(mol1) != waterMolecules.end()) ||
            (mol2 && waterMolecules.find(mol2) != waterMolecules.end()) ||
            (mol3 && waterMolecules.find(mol3) != waterMolecules.end());

        if (involvesWater)
            throw(UserInputException(
                std::format(
                    "A water type molecule is included in the angle list of "
                    "the topology file \"{}\" at entry number {}. Requesting "
                    "the use of the \"{}\" intramolecular water type model "
                    "expects the molecules of this moltype not to appear in "
                    "the topology file.",
                    FileSettings::getTopologyFileName(),
                    angleIndex,
                    string(WaterModelSettings::getWaterIntraModel())
                )
            ));

        ++angleIndex;
    }
}