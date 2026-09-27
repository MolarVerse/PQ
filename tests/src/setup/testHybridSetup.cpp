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

#include "atom.hpp"
#include "exceptions.hpp"
#include "hybridSetup.hpp"
#include "inputFileParser/hybridInputParser.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "qmSettings.hpp"
#include "settings.hpp"
#include "testSetup.hpp"

namespace
{
    void addSingleAtomMolecule(engine::Engine &engine, MolType molType)
    {
        auto atom = std::make_shared<molsys::Atom>();
        atom->setPosition({static_cast<double>(molType.get()), 0.0, 0.0});

        molsys::Molecule molecule;
        molecule.setMoltype(molType);
        molecule.addAtom(atom);

        engine.getSimulationBox().addAtom(atom);
        engine.getSimulationBox().addMolecule(molecule);
    }

    void configureValidHybridSettings(engine::Engine &engine)
    {
        settings::Settings::setJobtype(settings::JobType::QMMM_MD);
        settings::QMSettings::setQMMethod(settings::QMMethod::DFTBPLUS);
        settings::HybridSettings::setForcedCoreList({});
        settings::HybridSettings::setForcedLayerList({});
        settings::HybridSettings::setForcedOuterList({});
        settings::HybridSettings::setUseQMCharges(true);
        settings::HybridSettings::setCoreRadius(2.0);
        settings::HybridSettings::setLayerRadius(4.0);
        settings::HybridSettings::setSmoothingRegionThickness(1.0);
        settings::HybridSettings::setPointChargeThickness(2.0);
        engine.getSimulationBox().setBoxDimensions({40.0, 40.0, 40.0});
    }

}   // namespace

/* ---------- free function ---------- */

TEST_F(TestSetup, setupHybridIsNoOpWhenQMMMNotActive)
{
    settings::Settings::setJobtype(settings::JobType::MM_MD);   // not QMMM_MD
    EXPECT_NO_THROW(setup::setupHybrid(*_engine));
}

/* ---------- parseSelectionNoPython ---------- */

TEST_F(TestSetup, parseSelectionNoPythonSingleIndex)
{
    input::HybridInputParser parser;
    const auto               value =
        input::HybridInputParser::parseSelectionNoPython("3", "qm_center");
    ASSERT_EQ(value.size(), 1U);
    EXPECT_EQ(value[0], 3);
}

TEST_F(TestSetup, parseSelectionNoPythonCommaList)
{
    input::HybridInputParser parser;
    const auto               value =
        input::HybridInputParser::parseSelectionNoPython("1,3,5", "qm_center");
    ASSERT_EQ(value.size(), 3U);
    EXPECT_EQ(value[0], 1);
    EXPECT_EQ(value[1], 3);
    EXPECT_EQ(value[2], 5);
}

TEST_F(TestSetup, parseSelectionNoPythonRange)
{
    input::HybridInputParser parser;
    const auto               value =
        input::HybridInputParser::parseSelectionNoPython("2-5", "qm_center");
    ASSERT_EQ(value.size(), 4U);
    EXPECT_EQ(value[0], 2);
    EXPECT_EQ(value[3], 5);
}

TEST_F(TestSetup, parseSelectionNoPythonMixedRangeAndList)
{
    input::HybridInputParser parser;
    const auto value = input::HybridInputParser::parseSelectionNoPython(
        "1,3-4,7",
        "qm_center"
    );
    ASSERT_EQ(value.size(), 4U);
    EXPECT_EQ(value[0], 1);
    EXPECT_EQ(value[1], 3);
    EXPECT_EQ(value[2], 4);
    EXPECT_EQ(value[3], 7);
}

TEST_F(TestSetup, parseSelectionNoPythonEmptyThrows)
{
    input::HybridInputParser parser;
    EXPECT_THROW(
        parser.parseSelectionNoPython("", "qm_center"),
        exc::InputFileException
    );
}

/* ---------- parseSelection ---------- */

TEST_F(TestSetup, parseSelectionEmptyReturnsZeroOnly)
{
    input::HybridInputParser parser;
    const auto               value =
        input::HybridInputParser::parseSelection("", "qm_center");
    ASSERT_EQ(value.size(), 1U);
    EXPECT_EQ(value[0], 0);
}

TEST_F(TestSetup, parseSelectionSortsAndDeduplicates)
{
    input::HybridInputParser parser;
    const auto               value =
        input::HybridInputParser::parseSelection("5,1,3,1", "qm_center");
    ASSERT_EQ(value.size(), 3U);
    EXPECT_EQ(value[0], 1);
    EXPECT_EQ(value[1], 3);
    EXPECT_EQ(value[2], 5);
}

#ifndef PYTHON_ENABLED
TEST_F(TestSetup, parseSelectionWithLettersThrowsWithoutPython)
{
    input::HybridInputParser parser;
    EXPECT_THROW(
        parser.parseSelection("not_a_number", "qm_center"),
        exc::InputFileException
    );
}
#endif

/* ---------- setup throws ---------- */

TEST_F(TestSetup, setupThrowsNotImplemented)
{
    setup::HybridSetup hybridSetup{*_engine};
    EXPECT_THROW(hybridSetup.setup(), exc::InputFileException);
}

TEST_F(TestSetup, setupHybridConfiguresDefaultCenter)
{
    configureValidHybridSettings(*_engine);
    addSingleAtomMolecule(*_engine, MolType{1});

    EXPECT_NO_THROW(setup::setupHybrid(*_engine));
    EXPECT_EQ(
        _engine->getSimulationBox().getInnerRegionCenterAtomIndices(),
        std::vector<size_t>{0}
    );
}

TEST_F(TestSetup, setupHybridConfiguresExplicitLists)
{
    configureValidHybridSettings(*_engine);
    settings::QMSettings::setQMMethod(settings::QMMethod::TURBOMOLE);
    settings::HybridSettings::setInnerRegionCenter({0, 1});
    settings::HybridSettings::setForcedCoreList({0});
    settings::HybridSettings::setForcedLayerList({1});
    settings::HybridSettings::setForcedOuterList({2});
    settings::HybridSettings::setUseQMCharges(false);
    addSingleAtomMolecule(*_engine, MolType{1});
    addSingleAtomMolecule(*_engine, MolType{2});
    addSingleAtomMolecule(*_engine, MolType{3});

    EXPECT_NO_THROW(setup::HybridSetup{*_engine}.setup());
    EXPECT_TRUE(_engine->getSimulationBox().getMolecule(0).isForcedCore());
    EXPECT_TRUE(_engine->getSimulationBox().getMolecule(1).isForcedLayer());
    EXPECT_TRUE(_engine->getSimulationBox().getMolecule(2).isForcedOuter());
}

TEST_F(TestSetup, hybridSetupRejectsUnsupportedQmMethods)
{
    setup::HybridSetup   setup{*_engine};
    constexpr std::array unsupported{
        settings::QMMethod::PYSCF,
        settings::QMMethod::ASEDFTBPLUS,
        settings::QMMethod::ASEXTB,
        settings::QMMethod::MACE,
        settings::QMMethod::FENNOL,
        settings::QMMethod::NONE,
    };

    for (const auto method : unsupported)
    {
        settings::QMSettings::setQMMethod(method);
        EXPECT_THROW(setup.validateQMMethod(), exc::InputFileException);
    }

    settings::QMSettings::setQMMethod(settings::QMMethod::DFTBPLUS);
    EXPECT_NO_THROW(setup.validateQMMethod());
    settings::QMSettings::setQMMethod(settings::QMMethod::TURBOMOLE);
    EXPECT_NO_THROW(setup.validateQMMethod());
}

TEST_F(TestSetup, hybridSetupValidatesZoneRadii)
{
    _engine->getSimulationBox().setBoxDimensions({40.0, 40.0, 40.0});
    setup::HybridSetup setup{*_engine};

    settings::HybridSettings::setCoreRadius(5.0);
    settings::HybridSettings::setLayerRadius(4.0);
    settings::HybridSettings::setSmoothingRegionThickness(1.0);
    settings::HybridSettings::setPointChargeThickness(0.0);
    EXPECT_THROW(setup.checkZoneRadii(), exc::InputFileException);

    settings::HybridSettings::setCoreRadius(3.5);
    settings::HybridSettings::setLayerRadius(4.0);
    settings::HybridSettings::setSmoothingRegionThickness(1.0);
    EXPECT_THROW(setup.checkZoneRadii(), exc::InputFileException);

    settings::HybridSettings::setCoreRadius(2.0);
    settings::HybridSettings::setLayerRadius(11.0);
    settings::HybridSettings::setSmoothingRegionThickness(1.0);
    EXPECT_THROW(setup.checkZoneRadii(), exc::InputFileException);

    settings::HybridSettings::setCoreRadius(1.0);
    settings::HybridSettings::setLayerRadius(2.0);
    settings::HybridSettings::setSmoothingRegionThickness(0.5);
    settings::HybridSettings::setPointChargeThickness(59.0);
    EXPECT_THROW(setup.checkZoneRadii(), exc::InputFileException);

    settings::HybridSettings::setPointChargeThickness(2.0);
    EXPECT_NO_THROW(setup.checkZoneRadii());
}

TEST_F(TestSetup, hybridSetupRejectsMmChargesForMoltypeZero)
{
    _engine->getSimulationBox().addMoleculeType(
        molsys::MoleculeType(MolType{0})
    );
    settings::HybridSettings::setUseQMCharges(false);
    setup::HybridSetup setup{*_engine};

    EXPECT_THROW(setup.validateQMChargeSettings(), exc::InputFileException);

    settings::HybridSettings::setUseQMCharges(true);
    EXPECT_NO_THROW(setup.validateQMChargeSettings());
}
