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

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "angleForceField.hpp"
#include "atom.hpp"
#include "bondForceField.hpp"
#include "exceptions.hpp"
#include "interWater.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "settings.hpp"
#include "strongTypes.hpp"
#include "testSetup.hpp"
#include "throwWithMessage.hpp"
#include "waterModelSettings.hpp"
#include "waterModelSetup.hpp"

namespace
{
    constexpr MolType kWaterType{1};

    void addWaterSystem(
        engine::MDEngine               &engine,
        const std::vector<std::string> &atomNames
    )
    {
        auto &simBox = engine.getSimulationBox();
        simBox.setWaterType(kWaterType);

        molsys::MoleculeType waterType{kWaterType};
        waterType.setNumberOfAtoms(3);
        for (const auto &name : atomNames) waterType.addAtomName(name);
        simBox.addMoleculeType(waterType);

        molsys::Molecule water;
        water.setMoltype(kWaterType);

        for (size_t i = 0; i < 3; ++i)
        {
            auto atom = std::make_shared<molsys::Atom>();
            atom->setPartialCharge(0.0);
            atom->setPosition({static_cast<double>(i), 0.0, 0.0});
            water.addAtom(atom);
            simBox.addAtom(atom);
        }

        simBox.addMolecule(water);
    }

    void addWaterSystem(engine::MDEngine &engine)
    {
        addWaterSystem(engine, {"O", "H", "H"});
    }

    template <typename Parameter>
    void setupInterModel(
        engine::MDEngine               &engine,
        const settings::WaterInterModel model
    )
    {
        const auto state = waterModel::makeInterWaterState<Parameter>();
        engine.getSimulationBox().getMolecule(0).setPartialCharges(
            {state._oxygenCharge, state._hydrogenCharge, state._hydrogenCharge}
        );
        settings::WaterModelSettings::setWaterInterModel(model);
        setup::WaterModelSetup(engine).setup();
    }

    void configureNoInterModel()
    {
        settings::Settings::setJobtype(settings::JobType::MM_MD);
        settings::WaterModelSettings::setWaterInterModel(
            settings::WaterInterModel::NONE
        );
    }

}   // namespace

TEST_F(TestSetup, waterModelSetupCoversAllIntermolecularModels)
{
    configureNoInterModel();
    settings::WaterModelSettings::setWaterIntraModel(
        settings::WaterIntraModel::NONE
    );
    addWaterSystem(*_mdEngine);

    setupInterModel<waterModel::SPCInterParam>(
        *_mdEngine,
        settings::WaterInterModel::SPC
    );
    setupInterModel<waterModel::SPCEInterParam>(
        *_mdEngine,
        settings::WaterInterModel::SPC_E
    );
    setupInterModel<waterModel::SPCFwInterParam>(
        *_mdEngine,
        settings::WaterInterModel::SPC_FW
    );
    setupInterModel<waterModel::qSPCFwInterParam>(
        *_mdEngine,
        settings::WaterInterModel::QSPC_FW
    );
    setupInterModel<waterModel::SPCDCInterParam>(
        *_mdEngine,
        settings::WaterInterModel::SPC_DC
    );
    setupInterModel<waterModel::H2ODCInterParam>(
        *_mdEngine,
        settings::WaterInterModel::H2O_DC
    );
    setupInterModel<waterModel::TIP3PInterParam>(
        *_mdEngine,
        settings::WaterInterModel::TIP3P
    );
    setupInterModel<waterModel::OPC3InterParam>(
        *_mdEngine,
        settings::WaterInterModel::OPC3
    );
    setupInterModel<waterModel::SPCmTRInterParam>(
        *_mdEngine,
        settings::WaterInterModel::SPC_MTR
    );

    settings::Settings::activateCellList();
    setupInterModel<waterModel::TIP3PmTRInterParam>(
        *_mdEngine,
        settings::WaterInterModel::TIP3P_MTR
    );
}

TEST_F(TestSetup, waterModelSetupCoversAllIntramolecularModels)
{
    configureNoInterModel();
    addWaterSystem(*_mdEngine);

    constexpr std::array models{
        settings::WaterIntraModel::SPC,
        settings::WaterIntraModel::SPC_E,
        settings::WaterIntraModel::SPC_FW,
        settings::WaterIntraModel::QSPC_FW,
        settings::WaterIntraModel::SPC_DC,
        settings::WaterIntraModel::H2O_DC,
        settings::WaterIntraModel::TIP3P,
        settings::WaterIntraModel::OPC3,
        settings::WaterIntraModel::SPC_MTR,
        settings::WaterIntraModel::TIP3P_MTR,
        settings::WaterIntraModel::NONE,
    };

    settings::WaterModelSettings::setWaterIntraModel(models.front());
    setup::setupWaterModel(*_mdEngine);

    for (size_t i = 1; i < models.size(); ++i)
    {
        settings::WaterModelSettings::setWaterIntraModel(models.at(i));
        setup::WaterModelSetup(*_mdEngine).setup();
    }

    const auto &constraints = _mdEngine->getConstraints();
    EXPECT_TRUE(constraints->isShakeActive());
    EXPECT_EQ(constraints->getNumberOfBondConstraints(), 18);
}

TEST_F(TestSetup, waterModelSetupRejectsMissingWaterType)
{
    configureNoInterModel();
    settings::WaterModelSettings::setWaterIntraModel(
        settings::WaterIntraModel::NONE
    );
    EXPECT_THROW_MSG(

        setup::WaterModelSetup(*_mdEngine).setup(),

        exc::UserInputException,
        "Use of water model has been requested in the input file, but no water "
        "type is specified in the moldescriptor file."
    );
}

TEST_F(TestSetup, waterModelSetupRejectsInvalidAtomOrder)
{
    configureNoInterModel();
    settings::WaterModelSettings::setWaterIntraModel(
        settings::WaterIntraModel::NONE
    );
    addWaterSystem(*_mdEngine, {"H", "O", "H"});
    EXPECT_THROW_MSG(

        setup::WaterModelSetup(*_mdEngine).setup(),

        exc::MolDescriptorException,
        "Water molecule type must have exactly 3 atoms in the following order: "
        "O (oxygen), H (hydrogen), H (hydrogen)."
    );
}

TEST_F(TestSetup, waterModelSetupRejectsQmOnlyJobs)
{
    configureNoInterModel();
    settings::WaterModelSettings::setWaterIntraModel(
        settings::WaterIntraModel::NONE
    );
    addWaterSystem(*_mdEngine);
    settings::Settings::setJobtype(settings::JobType::QM_MD);
    EXPECT_THROW_MSG(

        setup::WaterModelSetup(*_mdEngine).setup(),

        exc::UserInputException,
        "Water models are not supported for QM-only job types."

    );
}

TEST_F(TestSetup, waterModelSetupRejectsMismatchedCharges)
{
    configureNoInterModel();
    settings::WaterModelSettings::setWaterIntraModel(
        settings::WaterIntraModel::NONE
    );
    settings::WaterModelSettings::setWaterInterModel(
        settings::WaterInterModel::SPC
    );
    addWaterSystem(*_mdEngine);
    EXPECT_THROW_MSG(

        setup::WaterModelSetup(*_mdEngine).setup(),

        exc::UserInputException,
        "Water molecule partial charge mismatch for atom O: expected -0.82 "
        "(according to SPC water model), got 0."
    );
}

TEST_F(TestSetup, waterModelSetupRejectsWaterBondsInTopology)
{
    configureNoInterModel();
    settings::WaterModelSettings::setWaterIntraModel(
        settings::WaterIntraModel::SPC_FW
    );
    addWaterSystem(*_mdEngine);
    auto *water = &_mdEngine->getSimulationBox().getMolecule(0);
    _mdEngine->getForceField()->addBond(
        ff::BondForceField(water, water, AtomIndex{0}, AtomIndex{1}, BondId{0})
    );

    EXPECT_THROW_MSG(

        setup::WaterModelSetup(*_mdEngine).setup(),

        exc::UserInputException,
        "A water type molecule is included in the bond list of the topology "
        "file \"\" at entry number 1. Requesting the use of the \"SPC_FW\" "
        "intramolecular water type model expects the molecules of this moltype "
        "not to appear in the topology file."
    );
}

TEST_F(TestSetup, waterModelSetupRejectsWaterAnglesInTopology)
{
    configureNoInterModel();
    settings::WaterModelSettings::setWaterIntraModel(
        settings::WaterIntraModel::SPC_FW
    );
    addWaterSystem(*_mdEngine);
    auto *water = &_mdEngine->getSimulationBox().getMolecule(0);
    _mdEngine->getForceField()->addAngle(
        ff::AngleForceField(
            {water, water, water},
            {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}},
            AngleId{0}
        )
    );

    EXPECT_THROW_MSG(

        setup::WaterModelSetup(*_mdEngine).setup(),

        exc::UserInputException,
        "A water type molecule is included in the angle list of the topology "
        "file \"\" at entry number 1. Requesting the use of the \"SPC_FW\" "
        "intramolecular water type model expects the molecules of this moltype "
        "not to appear in the topology file."
    );
}
