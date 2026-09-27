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
#include "throwWithMessage.hpp"

using namespace setup;
using namespace settings;
using namespace exc;
using namespace input;

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
        Settings::setJobtype(JobType::QMMM_MD);
        QMSettings::setQMMethod(QMMethod::DFTBPLUS);
        HybridSettings::setForcedCoreList({});
        HybridSettings::setForcedLayerList({});
        HybridSettings::setForcedOuterList({});
        HybridSettings::setUseQMCharges(true);
        HybridSettings::setCoreRadius(2.0);
        HybridSettings::setLayerRadius(4.0);
        HybridSettings::setSmoothingRegionThickness(1.0);
        HybridSettings::setPointChargeThickness(2.0);
        engine.getSimulationBox().setBoxDimensions({40.0, 40.0, 40.0});
    }

}   // namespace

/* ---------- free function ---------- */

TEST_F(TestSetup, setupHybridIsNoOpWhenQMMMNotActive)
{
    Settings::setJobtype(JobType::MM_MD);   // not QMMM_MD
    EXPECT_NO_THROW(setupHybrid(*_engine));
}

/* ---------- parseSelectionNoPython ---------- */

TEST_F(TestSetup, parseSelectionNoPythonSingleIndex)
{
    HybridInputParser parser;
    const auto        value =
        input::HybridInputParser::parseSelectionNoPython("3", "qm_center");
    ASSERT_EQ(value.size(), 1U);
    EXPECT_EQ(value[0], 3);
}

TEST_F(TestSetup, parseSelectionNoPythonCommaList)
{
    HybridInputParser parser;
    const auto        value =
        input::HybridInputParser::parseSelectionNoPython("1,3,5", "qm_center");
    ASSERT_EQ(value.size(), 3U);
    EXPECT_EQ(value[0], 1);
    EXPECT_EQ(value[1], 3);
    EXPECT_EQ(value[2], 5);
}

TEST_F(TestSetup, parseSelectionNoPythonRange)
{
    HybridInputParser parser;
    const auto        value =
        input::HybridInputParser::parseSelectionNoPython("2-5", "qm_center");
    ASSERT_EQ(value.size(), 4U);
    EXPECT_EQ(value[0], 2);
    EXPECT_EQ(value[3], 5);
}

TEST_F(TestSetup, parseSelectionNoPythonMixedRangeAndList)
{
    HybridInputParser parser;
    const auto        value = input::HybridInputParser::parseSelectionNoPython(
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
    HybridInputParser parser;
    EXPECT_THROW_MSG(
        parser.parseSelectionNoPython("", "qm_center"),
        InputFileException,
        "The value of key qm_center -  is an empty list. The qm_center string "
        "must be a comma-separated list of integers or ranges, representing "
        "the atom indices in the restart file that should be treated as the "
        "qm_center."
    );
}

/* ---------- parseSelection ---------- */

TEST_F(TestSetup, parseSelectionEmptyReturnsZeroOnly)
{
    HybridInputParser parser;
    const auto        value =
        input::HybridInputParser::parseSelection("", "qm_center");
    ASSERT_EQ(value.size(), 1U);
    EXPECT_EQ(value[0], 0);
}

TEST_F(TestSetup, parseSelectionSortsAndDeduplicates)
{
    HybridInputParser parser;
    const auto        value =
        input::HybridInputParser::parseSelection("5,1,3,1", "qm_center");
    ASSERT_EQ(value.size(), 3U);
    EXPECT_EQ(value[0], 1);
    EXPECT_EQ(value[1], 3);
    EXPECT_EQ(value[2], 5);
}

#ifndef PYTHON_ENABLED
TEST_F(TestSetup, parseSelectionWithLettersThrowsWithoutPython)
{
    HybridInputParser parser;
    EXPECT_THROW_MSG(
        parser.parseSelection("not_a_number", "qm_center"),
        InputFileException,
        "The value of key qm_center - not_a_number contains characters that "
        "are not digits, \"-\" or commas. The current build of PQ was compiled "
        "without Python bindings, so the qm_center string must be a "
        "comma-separated list of integers, representing the atom indices in "
        "the restart file that should be treated as the qm_center. In order to "
        "use the full selection parser power of the PQAnalysis Python package, "
        "the PQ build must be compiled with Python bindings."
    );
}
#endif

/* ---------- setup throws ---------- */

TEST_F(TestSetup, setupThrowsNotImplemented)
{
    HybridSetup hybridSetup{*_engine};
    EXPECT_THROW_MSG(
        hybridSetup.setup(),
        InputFileException,
        "QM method \"none\" is not supported for hybrid type calculations. "
        "Supported QM methods are \"dftbplus\" and \"turbomole\"."
    );
}

TEST_F(TestSetup, setupHybridConfiguresDefaultCenter)
{
    configureValidHybridSettings(*_engine);
    addSingleAtomMolecule(*_engine, MolType{1});

    EXPECT_NO_THROW(setupHybrid(*_engine));
    EXPECT_EQ(
        _engine->getSimulationBox().getInnerRegionCenterAtomIndices(),
        std::vector<size_t>{0}
    );
}

TEST_F(TestSetup, setupHybridConfiguresExplicitLists)
{
    configureValidHybridSettings(*_engine);
    QMSettings::setQMMethod(QMMethod::TURBOMOLE);
    HybridSettings::setInnerRegionCenter({0, 1});
    HybridSettings::setForcedCoreList({0});
    HybridSettings::setForcedLayerList({1});
    HybridSettings::setForcedOuterList({2});
    HybridSettings::setUseQMCharges(false);
    addSingleAtomMolecule(*_engine, MolType{1});
    addSingleAtomMolecule(*_engine, MolType{2});
    addSingleAtomMolecule(*_engine, MolType{3});

    EXPECT_NO_THROW(HybridSetup{*_engine}.setup());
    EXPECT_TRUE(_engine->getSimulationBox().getMolecule(0).isForcedCore());
    EXPECT_TRUE(_engine->getSimulationBox().getMolecule(1).isForcedLayer());
    EXPECT_TRUE(_engine->getSimulationBox().getMolecule(2).isForcedOuter());
}

TEST_F(TestSetup, hybridSetupRejectsUnsupportedQmMethods)
{
    HybridSetup          setup{*_engine};
    constexpr std::array unsupported{
        QMMethod::PYSCF,
        QMMethod::ASEDFTBPLUS,
        QMMethod::ASEXTB,
        QMMethod::MACE,
        QMMethod::FENNOL,
        QMMethod::NONE,
    };

    for (const auto method : unsupported)
    {
        QMSettings::setQMMethod(method);
        EXPECT_THROW_MSG(
            setup.validateQMMethod(),
            InputFileException,
            "QM method \"" + string(method) +
                "\" is not supported for hybrid type "
                "calculations. Supported QM methods are \"dftbplus\" and "
                "\"turbomole\"."
        );
    }

    QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    EXPECT_NO_THROW(setup.validateQMMethod());
    QMSettings::setQMMethod(QMMethod::TURBOMOLE);
    EXPECT_NO_THROW(setup.validateQMMethod());
}

TEST_F(TestSetup, hybridSetupValidatesZoneRadii)
{
    _engine->getSimulationBox().setBoxDimensions({40.0, 40.0, 40.0});
    HybridSetup setup{*_engine};

    HybridSettings::setCoreRadius(5.0);
    HybridSettings::setLayerRadius(4.0);
    HybridSettings::setSmoothingRegionThickness(1.0);
    HybridSettings::setPointChargeThickness(0.0);
    EXPECT_THROW_MSG(
        setup.checkZoneRadii(),
        InputFileException,
        "Core radius (5 Å) cannot be larger than layer radius (4 Å)"
    );

    HybridSettings::setCoreRadius(3.5);
    HybridSettings::setLayerRadius(4.0);
    HybridSettings::setSmoothingRegionThickness(1.0);
    EXPECT_THROW_MSG(
        setup.checkZoneRadii(),
        InputFileException,
        "Smoothing region is too thick (1 Å) for the chosen combination of "
        "core (3.5 Å) and layer radius (4 Å)"
    );

    HybridSettings::setCoreRadius(2.0);
    HybridSettings::setLayerRadius(11.0);
    HybridSettings::setSmoothingRegionThickness(1.0);
    EXPECT_THROW_MSG(
        setup.checkZoneRadii(),
        InputFileException,
        "Layer radius (11 Å) exceeds one quarter of the smallest box dimension "
        "(40 Å). This configuration is not allowed to ensure compliance with "
        "the minimum image convention."
    );

    HybridSettings::setCoreRadius(1.0);
    HybridSettings::setLayerRadius(2.0);
    HybridSettings::setSmoothingRegionThickness(0.5);
    HybridSettings::setPointChargeThickness(59.0);
    EXPECT_THROW_MSG(
        setup.checkZoneRadii(),
        InputFileException,
        "Layer radius (2 Å) plus point charge thickness (59 Å) exceeds three "
        "halves of the smallest box dimension (40 Å). This configuration is "
        "not allowed, as it would include point charges from beyond the "
        "immediate neighboring cells."
    );

    HybridSettings::setPointChargeThickness(2.0);
    EXPECT_NO_THROW(setup.checkZoneRadii());
}

TEST_F(TestSetup, hybridSetupRejectsMmChargesForMoltypeZero)
{
    _engine->getSimulationBox().addMoleculeType(
        molsys::MoleculeType(MolType{0})
    );
    HybridSettings::setUseQMCharges(false);
    HybridSetup setup{*_engine};

    EXPECT_THROW_MSG(
        setup.validateQMChargeSettings(),
        InputFileException,
        "Invalid configuration: MM charges requested (qm_charges = mm) in "
        "input file but atoms with moltype \"0\" are present in the system. "
        "Either set \"qm_charges = qm\" or ensure all atoms have anon-zero "
        "moltype."
    );

    HybridSettings::setUseQMCharges(true);
    EXPECT_NO_THROW(setup.validateQMChargeSettings());
}
