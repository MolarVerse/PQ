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

#include <cmath>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include "bondConstraint.hpp"     // for BondConstraint
#include "engine.hpp"             // for Engine
#include "exceptions.hpp"         // for customException
#include "fileSettings.hpp"       // for FileSettings
#include "interWater.hpp"         // for InterWater
#include "mdEngine.hpp"           // for MDEngine
#include "references.hpp"         // for References
#include "referencesOutput.hpp"   // for ReferencesOutput
#include "rigidWaterGeometry.hpp"
#include "settings.hpp"
#include "waterModelSettings.hpp"   // for WaterModelSettings

using namespace constants;
using namespace constraints;
using namespace customException;
using namespace engine;
using namespace references;
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
    if (water.getAtomNames() != std::vector<std::string>{"O", "H", "H"})
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

    addReferences();
}

/**
 * @brief Verify that water molecules do not appear in the topology file
 * bonds/angles.
 *
 * @details For intramolecular water models that use constraints, water
 * molecules should not have their bonds or angles defined in the topology file.
 * This function checks the bond and angle lists and throws an exception if any
 * water molecule is found.
 *
 * @throws UserInputException If a water type molecule is found in the bond or
 * angle list of the topology file.
 */
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
 * @brief Validate water molecule partial charges against the inter-water model.
 *
 * @throws UserInputException If any water molecule partial charge differs from
 * the expected inter-water model charge by more than the allowed tolerance.
 *
 * @param state Inter-water parameters providing expected charge values.
 */
void WaterModelSetup::checkMoldescriptorWaterCharge(
    const InterWaterState &state
)
{
    const auto modelName   = string(WaterModelSettings::getWaterInterModel());
    const auto checkCharge = [&modelName](
                                 const pq::Molecule &water,
                                 const size_t        atomIndex,
                                 const double        expected,
                                 const std::string  &atomName
                             )
    {
        constexpr double tol    = 1e-8;
        const auto       actual = water.getPartialCharge(atomIndex);
        if (std::abs(actual - expected) > tol)
            throw(UserInputException(
                std::format(
                    "Water molecule partial charge mismatch for atom {}: "
                    "expected {} (according to {} water model), got {}.",
                    atomName,
                    expected,
                    modelName,
                    actual
                )
            ));
    };

    const auto &waterMolecules =
        _engine.getSimulationBoxPtr()->getWaterTypeMolecules();

    for (const auto &water : waterMolecules)
    {
        checkCharge(water, 0, state._oxygenCharge, "O");
        checkCharge(water, 1, state._hydrogenCharge, "H1");
        checkCharge(water, 2, state._hydrogenCharge, "H2");
    };
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

    // clang-format off
    switch (intraModel)
    {
        case SPC: return RigidWaterGeometry{_SPC_OH_DIST_, _SPC_HH_DIST_};
        case SPC_E: return RigidWaterGeometry{_SPC_E_OH_DIST_, _SPC_E_HH_DIST_};
        case SPC_DC: return RigidWaterGeometry{_SPC_DC_OH_DIST_, _SPC_DC_HH_DIST_};
        case H2O_DC: return RigidWaterGeometry{_H2O_DC_OH_DIST_, _H2O_DC_HH_DIST_};
        case TIP3P: return RigidWaterGeometry{_TIP3P_OH_DIST_, _TIP3P_HH_DIST_};
        case OPC3: return RigidWaterGeometry{_OPC3_OH_DIST_, _OPC3_HH_DIST_};
        case SPC_FW:
        case QSPC_FW:
        case SPC_MTR:
        case TIP3P_MTR:
        case NONE: 
            return std::nullopt;
    }
    // clang-format on
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

    _engine.getConstraints().activateShake();
}

/**
 * @brief Set up the intermolecular water interaction model.
 *
 * @details Creates an @ref InterWater object with the appropriate parameters
 * and strategy (cell-list or brute-force) based on the configured
 * intermolecular water model. Also validates water molecule partial charges
 * against expected values.
 *
 * @throws UserInputException If water molecule partial charges do not match the
 * expected values for the selected intermolecular model.
 */
void WaterModelSetup::makeInterWater()
{
    using enum WaterInterModel;
    auto state = InterWaterState();

    const auto model = WaterModelSettings::getWaterInterModel();

    switch (model)
    {
        case SPC: state = makeInterWaterState<SPCInterParam>(); break;
        case SPC_E: state = makeInterWaterState<SPCEInterParam>(); break;
        case SPC_FW: state = makeInterWaterState<SPCFwInterParam>(); break;
        case QSPC_FW: state = makeInterWaterState<qSPCFwInterParam>(); break;
        case SPC_DC: state = makeInterWaterState<SPCDCInterParam>(); break;
        case H2O_DC: state = makeInterWaterState<H2ODCInterParam>(); break;
        case TIP3P: state = makeInterWaterState<TIP3PInterParam>(); break;
        case OPC3: state = makeInterWaterState<OPC3InterParam>(); break;
        case SPC_MTR: state = makeInterWaterState<SPCmTRInterParam>(); break;
        case TIP3P_MTR: state = makeInterWaterState<SPCmTRInterParam>(); break;
        case NONE: break;
    }

    checkMoldescriptorWaterCharge(state);

    std::unique_ptr<InterWaterStrategy> strategy;

    auto isCellListActivated = _engine.getCellList().isActive();
    if (isCellListActivated)
        strategy = std::make_unique<InterWaterStrategyCellList>();
    else
        strategy = std::make_unique<InterWaterStrategyBruteForce>();

    auto interWater =
        std::make_unique<InterWater>(std::move(state), std::move(strategy));

    _engine.setInterWater(std::move(interWater));
}

/**
 * @brief Add reference file entries for the configured water models.
 *
 * @details Adds bibliography references for both intramolecular and
 * intermolecular water models to the references output.
 */
void WaterModelSetup::addReferences()
{
    const auto intraModel = WaterModelSettings::getWaterIntraModel();
    const auto interModel = WaterModelSettings::getWaterInterModel();

    // clang-format off
    switch (intraModel)
    {
        using enum WaterIntraModel;
        case SPC: ReferencesOutput::addReferenceFile(_SPC_FILE_); break;
        case SPC_E: ReferencesOutput::addReferenceFile(_SPC_E_FILE_); break;
        case SPC_FW: ReferencesOutput::addReferenceFile(_SPC_FW_FILE_); break;
        case QSPC_FW: ReferencesOutput::addReferenceFile(_QSPC_FW_FILE_); break;
        case SPC_DC: ReferencesOutput::addReferenceFile(_SPC_DC_FILE_); break;
        case H2O_DC: ReferencesOutput::addReferenceFile(_H2O_DC_FILE_); break;
        case TIP3P: ReferencesOutput::addReferenceFile(_TIP3P_FILE_); break;
        case OPC3: ReferencesOutput::addReferenceFile(_OPC3_FILE_); break;
        case SPC_MTR: ReferencesOutput::addReferenceFile(_SPC_MTR_FILE_); break;
        case TIP3P_MTR: ReferencesOutput::addReferenceFile(_TIP3P_MTR_FILE_); break;
        case NONE: break;
    }

    switch (interModel)
    {
        using enum WaterInterModel;
        case SPC: ReferencesOutput::addReferenceFile(_SPC_FILE_); break;
        case SPC_E: ReferencesOutput::addReferenceFile(_SPC_E_FILE_); break;
        case SPC_FW: ReferencesOutput::addReferenceFile(_SPC_FW_FILE_); break;
        case QSPC_FW: ReferencesOutput::addReferenceFile(_QSPC_FW_FILE_); break;
        case SPC_DC: ReferencesOutput::addReferenceFile(_SPC_DC_FILE_); break;
        case H2O_DC: ReferencesOutput::addReferenceFile(_H2O_DC_FILE_); break;
        case TIP3P: ReferencesOutput::addReferenceFile(_TIP3P_FILE_); break;
        case OPC3: ReferencesOutput::addReferenceFile(_OPC3_FILE_); break;
        case SPC_MTR: ReferencesOutput::addReferenceFile(_SPC_MTR_FILE_); break;
        case TIP3P_MTR: ReferencesOutput::addReferenceFile(_TIP3P_MTR_FILE_); break;
        case NONE: break;
    }
    //clang-format on
}