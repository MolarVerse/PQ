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
#include <optional>
#include <unordered_set>

#include "bondConstraint.hpp"       // for BondConstraint
#include "engine.hpp"               // for Engine
#include "exceptions.hpp"           // for customException
#include "fileSettings.hpp"         // for FileSettings
#include "interWater.hpp"           // for InterWater
#include "mdEngine.hpp"             // for MDEngine
#include "molecule.hpp"             // for Molecule
#include "waterModelSettings.hpp"   // for WaterModelSettings

using namespace constraints;
using namespace customException;
using namespace engine;
using namespace settings;
using namespace setup;
using namespace waterModel;

/**
 * @brief Set up the configured water model for an MD engine.
 *
 * @details This is the public entry point to the water model setup. It writes
 * setup output, constructs a @ref WaterModelSetup helper and runs the actual
 * setup routine.
 *
 * @param engine The engine that should receive the water model setup.
 */
void setup::setupWaterModel(Engine &engine)
{
    engine.getStdoutOutput().writeSetup("Water model");
    engine.getLogOutput().writeSetup("Water model");

    WaterModelSetup waterModelSetup(dynamic_cast<MDEngine &>(engine));
    waterModelSetup.setup();
}

/**
 * @brief Construct a water model setup helper.
 *
 * @param engine The MD engine used to access simulation state, constraints,
 * and force-field data during water model setup.
 */
WaterModelSetup::WaterModelSetup(MDEngine &engine) : _engine(engine) {}

/**
 * @brief Perform all water model setup steps.
 *
 * @details Validates that a water moltype exists, checks that the molecule
 * descriptor matches the supported geometry, rejects unsupported job types,
 * optionally verifies the topology file, and installs constraints for rigid
 * water models.
 *
 * @throws UserInputException If water model setup is requested without a water
 * moltype or for unsupported QM-only jobs.
 * @throws MolDescriptorException If the selected water molecule does not
 * contain exactly three atoms in O-H-H order.
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

    if (WaterModelSettings::getWaterIntraModel() != WaterIntraModel::NONE)
        checkTopologyFile();

    if (const auto geometry =
            getRigidWaterGeometry(WaterModelSettings::getWaterIntraModel());
        geometry.has_value())
        shakeSetupForRigidWater(geometry.value());

    if (WaterModelSettings::getWaterInterModel() != WaterInterModel::NONE)
        makeInterWater();

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

/**
 * @brief Get the rigid geometry parameters for a water model.
 *
 * @details Only rigid water variants return geometry data. Flexible water
 * models return @c std::nullopt and are handled without constraints.
 *
 * @param intraModel The configured intramolecular water model.
 * @return The O-H and H-H distances for rigid models, or @c std::nullopt if
 * the model is not rigid.
 */
std::optional<RigidWaterGeometry> WaterModelSetup::getRigidWaterGeometry(
    const WaterIntraModel intraModel
)
{
    using enum WaterIntraModel;

    switch (intraModel)
    {
        case TIP3P: return RigidWaterGeometry{0.9572, 1.5139};
        case OPC3: return RigidWaterGeometry{0.97888, 1.598492306};
        default: return std::nullopt;
    }
}

/**
 * @brief Construct and add SHAKE-style bond constraints for rigid water
 * molecules.
 *
 * @details Adds constraints for both O-H bonds and the H-H distance of each
 * water type molecule in the current simulation box.
 *
 * @param geometry The target rigid water geometry.
 */
void WaterModelSetup::shakeSetupForRigidWater(
    const RigidWaterGeometry &geometry
)
{
    const auto   dOH     = geometry.dOH;
    const auto   dHH     = geometry.dHH;
    const size_t OIndex  = 0;
    const size_t H1Index = 1;
    const size_t H2Index = 2;

    for (auto &waterMol : _engine.getSimulationBox().getWaterTypeMolecules())
    {
        auto bondConstraintOH1 =
            BondConstraint(&waterMol, &waterMol, OIndex, H1Index, dOH);
        auto bondConstraintOH2 =
            BondConstraint(&waterMol, &waterMol, OIndex, H2Index, dOH);
        auto bondConstraintHH =
            BondConstraint(&waterMol, &waterMol, H1Index, H2Index, dHH);

        _engine.getConstraints().addBondConstraint(bondConstraintOH1);
        _engine.getConstraints().addBondConstraint(bondConstraintOH2);
        _engine.getConstraints().addBondConstraint(bondConstraintHH);
    }
}

void WaterModelSetup::makeInterWater()
{
    using enum WaterInterModel;
    auto state = InterWaterState();

    const auto model = WaterModelSettings::getWaterInterModel();

    switch (model)
    {
        case SPC_FW: state = makeInterWaterState<SPCFwInterParam>(); break;
        case QSPC_FW: state = makeInterWaterState<qSPCFwInterParam>(); break;
        case TIP3P: state = makeInterWaterState<TIP3PInterParam>(); break;
        case OPC3: state = makeInterWaterState<OPC3InterParam>(); break;
        default: break;
    }

    std::unique_ptr<InterWaterStrategy> strategy;

    auto isCellListActivated = _engine.getCellList().isActive();
    if (isCellListActivated)
        strategy = std::make_unique<InterWaterStrategyBruteForce>();
    // TODO: implement cell-list strategy
    else
        strategy = std::make_unique<InterWaterStrategyBruteForce>();

    auto interWater = std::make_unique<InterWater>(state, std::move(strategy));

    _engine.setInterWater(std::move(interWater));
}