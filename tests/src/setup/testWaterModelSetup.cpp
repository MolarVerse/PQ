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
#include "testSetup.hpp"
#include "waterModelSettings.hpp"
#include "waterModelSetup.hpp"

using customException::MolDescriptorException;
using customException::UserInputException;
using settings::JobType;
using settings::Settings;
using settings::WaterInterModel;
using settings::WaterIntraModel;
using settings::WaterModelSettings;
using setup::WaterModelSetup;
using simulationBox::Atom;
using simulationBox::Molecule;
using simulationBox::MoleculeType;

namespace
{
    constexpr size_t kWaterType = 1;

    void addWaterSystem(
        engine::MDEngine               &engine,
        const std::vector<std::string> &atomNames
    )
    {
        auto &simBox = engine.getSimulationBox();
        simBox.setWaterType(kWaterType);

        MoleculeType waterType(kWaterType);
        waterType.setNumberOfAtoms(3);
        for (const auto &name : atomNames) waterType.addAtomName(name);
        simBox.addMoleculeType(waterType);

        Molecule water;
        water.setMoltype(kWaterType);
        water.setNumberOfAtoms(3);

        for (size_t i = 0; i < 3; ++i)
        {
            auto atom = std::make_shared<Atom>();
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
    void setupInterModel(engine::MDEngine &engine, const WaterInterModel model)
    {
        const auto state = waterModel::makeInterWaterState<Parameter>();
        engine.getSimulationBox().getMolecule(0).setPartialCharges(
            {state._oxygenCharge, state._hydrogenCharge, state._hydrogenCharge}
        );
        WaterModelSettings::setWaterInterModel(model);
        WaterModelSetup(engine).setup();
    }

    void configureNoInterModel()
    {
        Settings::setJobtype(JobType::MM_MD);
        WaterModelSettings::setWaterInterModel(WaterInterModel::NONE);
    }

}   // namespace

TEST_F(TestSetup, waterModelSetupCoversAllIntermolecularModels)
{
    configureNoInterModel();
    WaterModelSettings::setWaterIntraModel(WaterIntraModel::NONE);
    addWaterSystem(*_mdEngine);

    setupInterModel<waterModel::SPCInterParam>(
        *_mdEngine,
        WaterInterModel::SPC
    );
    setupInterModel<waterModel::SPCEInterParam>(
        *_mdEngine,
        WaterInterModel::SPC_E
    );
    setupInterModel<waterModel::SPCFwInterParam>(
        *_mdEngine,
        WaterInterModel::SPC_FW
    );
    setupInterModel<waterModel::qSPCFwInterParam>(
        *_mdEngine,
        WaterInterModel::QSPC_FW
    );
    setupInterModel<waterModel::SPCDCInterParam>(
        *_mdEngine,
        WaterInterModel::SPC_DC
    );
    setupInterModel<waterModel::H2ODCInterParam>(
        *_mdEngine,
        WaterInterModel::H2O_DC
    );
    setupInterModel<waterModel::TIP3PInterParam>(
        *_mdEngine,
        WaterInterModel::TIP3P
    );
    setupInterModel<waterModel::OPC3InterParam>(
        *_mdEngine,
        WaterInterModel::OPC3
    );
    setupInterModel<waterModel::SPCmTRInterParam>(
        *_mdEngine,
        WaterInterModel::SPC_MTR
    );

    settings::Settings::activateCellList();
    setupInterModel<waterModel::TIP3PmTRInterParam>(
        *_mdEngine,
        WaterInterModel::TIP3P_MTR
    );
}

TEST_F(TestSetup, waterModelSetupCoversAllIntramolecularModels)
{
    configureNoInterModel();
    addWaterSystem(*_mdEngine);

    constexpr std::array models{
        WaterIntraModel::SPC,
        WaterIntraModel::SPC_E,
        WaterIntraModel::SPC_FW,
        WaterIntraModel::QSPC_FW,
        WaterIntraModel::SPC_DC,
        WaterIntraModel::H2O_DC,
        WaterIntraModel::TIP3P,
        WaterIntraModel::OPC3,
        WaterIntraModel::SPC_MTR,
        WaterIntraModel::TIP3P_MTR,
        WaterIntraModel::NONE,
    };

    WaterModelSettings::setWaterIntraModel(models.front());
    setup::setupWaterModel(*_mdEngine);

    for (size_t i = 1; i < models.size(); ++i)
    {
        WaterModelSettings::setWaterIntraModel(models.at(i));
        WaterModelSetup(*_mdEngine).setup();
    }

    const auto &constraints = _mdEngine->getConstraints();
    EXPECT_TRUE(constraints->isShakeActive());
    EXPECT_EQ(constraints->getNumberOfBondConstraints(), 18);
}

TEST_F(TestSetup, waterModelSetupRejectsMissingWaterType)
{
    configureNoInterModel();
    WaterModelSettings::setWaterIntraModel(WaterIntraModel::NONE);
    EXPECT_THROW(WaterModelSetup(*_mdEngine).setup(), UserInputException);
}

TEST_F(TestSetup, waterModelSetupRejectsInvalidAtomOrder)
{
    configureNoInterModel();
    WaterModelSettings::setWaterIntraModel(WaterIntraModel::NONE);
    addWaterSystem(*_mdEngine, {"H", "O", "H"});
    EXPECT_THROW(WaterModelSetup(*_mdEngine).setup(), MolDescriptorException);
}

TEST_F(TestSetup, waterModelSetupRejectsQmOnlyJobs)
{
    configureNoInterModel();
    WaterModelSettings::setWaterIntraModel(WaterIntraModel::NONE);
    addWaterSystem(*_mdEngine);
    Settings::setJobtype(JobType::QM_MD);
    EXPECT_THROW(WaterModelSetup(*_mdEngine).setup(), UserInputException);
}

TEST_F(TestSetup, waterModelSetupRejectsMismatchedCharges)
{
    configureNoInterModel();
    WaterModelSettings::setWaterIntraModel(WaterIntraModel::NONE);
    WaterModelSettings::setWaterInterModel(WaterInterModel::SPC);
    addWaterSystem(*_mdEngine);
    EXPECT_THROW(WaterModelSetup(*_mdEngine).setup(), UserInputException);
}

TEST_F(TestSetup, waterModelSetupRejectsWaterBondsInTopology)
{
    configureNoInterModel();
    WaterModelSettings::setWaterIntraModel(WaterIntraModel::SPC_FW);
    addWaterSystem(*_mdEngine);
    auto *water = &_mdEngine->getSimulationBox().getMolecule(0);
    _mdEngine->getForceField()->addBond(
        forceField::BondForceField(water, water, 0, 1, 0)
    );

    EXPECT_THROW(WaterModelSetup(*_mdEngine).setup(), UserInputException);
}

TEST_F(TestSetup, waterModelSetupRejectsWaterAnglesInTopology)
{
    configureNoInterModel();
    WaterModelSettings::setWaterIntraModel(WaterIntraModel::SPC_FW);
    addWaterSystem(*_mdEngine);
    auto *water = &_mdEngine->getSimulationBox().getMolecule(0);
    _mdEngine->getForceField()->addAngle(
        forceField::AngleForceField({water, water, water}, {0, 1, 2}, 0)
    );

    EXPECT_THROW(WaterModelSetup(*_mdEngine).setup(), UserInputException);
}
