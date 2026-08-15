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

using namespace setup;
using namespace settings;
using namespace customException;
using namespace input;

namespace
{
    void addSingleAtomMolecule(engine::Engine &engine, const size_t molType)
    {
        auto atom = std::make_shared<molsys::Atom>();
        atom->setPosition({static_cast<double>(molType), 0.0, 0.0});

        molsys::Molecule molecule;
        molecule.setMoltype(molType);
        molecule.setNumberOfAtoms(1);
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
    HybridInputParser parser(*_engine);
    const auto        v = parser.parseSelectionNoPython("3", "qm_center");
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0], 3);
}

TEST_F(TestSetup, parseSelectionNoPythonCommaList)
{
    HybridInputParser parser(*_engine);
    const auto        v = parser.parseSelectionNoPython("1,3,5", "qm_center");
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 3);
    EXPECT_EQ(v[2], 5);
}

TEST_F(TestSetup, parseSelectionNoPythonRange)
{
    HybridInputParser parser(*_engine);
    const auto        v = parser.parseSelectionNoPython("2-5", "qm_center");
    ASSERT_EQ(v.size(), 4u);
    EXPECT_EQ(v[0], 2);
    EXPECT_EQ(v[3], 5);
}

TEST_F(TestSetup, parseSelectionNoPythonMixedRangeAndList)
{
    HybridInputParser parser(*_engine);
    const auto        v = parser.parseSelectionNoPython("1,3-4,7", "qm_center");
    ASSERT_EQ(v.size(), 4u);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 3);
    EXPECT_EQ(v[2], 4);
    EXPECT_EQ(v[3], 7);
}

TEST_F(TestSetup, parseSelectionNoPythonEmptyThrows)
{
    HybridInputParser parser(*_engine);
    EXPECT_THROW(
        parser.parseSelectionNoPython("", "qm_center"),
        InputFileException
    );
}

/* ---------- parseSelection ---------- */

TEST_F(TestSetup, parseSelectionEmptyReturnsZeroOnly)
{
    HybridInputParser parser(*_engine);
    const auto        v = parser.parseSelection("", "qm_center");
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0], 0);
}

TEST_F(TestSetup, parseSelectionSortsAndDeduplicates)
{
    HybridInputParser parser(*_engine);
    const auto        v = parser.parseSelection("5,1,3,1", "qm_center");
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 3);
    EXPECT_EQ(v[2], 5);
}

#ifndef PYTHON_ENABLED
TEST_F(TestSetup, parseSelectionWithLettersThrowsWithoutPython)
{
    HybridInputParser parser(*_engine);
    EXPECT_THROW(
        parser.parseSelection("not_a_number", "qm_center"),
        InputFileException
    );
}
#endif

/* ---------- setup throws ---------- */

TEST_F(TestSetup, setupThrowsNotImplemented)
{
    HybridSetup hs(*_engine);
    EXPECT_THROW(hs.setup(), InputFileException);
}

TEST_F(TestSetup, setupHybridConfiguresDefaultCenter)
{
    configureValidHybridSettings(*_engine);
    addSingleAtomMolecule(*_engine, 1);

    EXPECT_NO_THROW(setupHybrid(*_engine));
    EXPECT_EQ(
        _engine->getSimulationBox().getInnerRegionCenterAtomIndices(),
        std::vector<int>{0}
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
    addSingleAtomMolecule(*_engine, 1);
    addSingleAtomMolecule(*_engine, 2);
    addSingleAtomMolecule(*_engine, 3);

    EXPECT_NO_THROW(HybridSetup(*_engine).setup());
    EXPECT_TRUE(_engine->getSimulationBox().getMolecule(0).isForcedCore());
    EXPECT_TRUE(_engine->getSimulationBox().getMolecule(1).isForcedLayer());
    EXPECT_TRUE(_engine->getSimulationBox().getMolecule(2).isForcedOuter());
}

TEST_F(TestSetup, hybridSetupRejectsUnsupportedQmMethods)
{
    HybridSetup          setup(*_engine);
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
        EXPECT_THROW(setup.validateQMMethod(), InputFileException);
    }

    QMSettings::setQMMethod(QMMethod::DFTBPLUS);
    EXPECT_NO_THROW(setup.validateQMMethod());
    QMSettings::setQMMethod(QMMethod::TURBOMOLE);
    EXPECT_NO_THROW(setup.validateQMMethod());
}

TEST_F(TestSetup, hybridSetupValidatesZoneRadii)
{
    _engine->getSimulationBox().setBoxDimensions({40.0, 40.0, 40.0});
    HybridSetup setup(*_engine);

    HybridSettings::setCoreRadius(5.0);
    HybridSettings::setLayerRadius(4.0);
    HybridSettings::setSmoothingRegionThickness(1.0);
    HybridSettings::setPointChargeThickness(0.0);
    EXPECT_THROW(setup.checkZoneRadii(), InputFileException);

    HybridSettings::setCoreRadius(3.5);
    HybridSettings::setLayerRadius(4.0);
    HybridSettings::setSmoothingRegionThickness(1.0);
    EXPECT_THROW(setup.checkZoneRadii(), InputFileException);

    HybridSettings::setCoreRadius(2.0);
    HybridSettings::setLayerRadius(11.0);
    HybridSettings::setSmoothingRegionThickness(1.0);
    EXPECT_THROW(setup.checkZoneRadii(), InputFileException);

    HybridSettings::setCoreRadius(1.0);
    HybridSettings::setLayerRadius(2.0);
    HybridSettings::setSmoothingRegionThickness(0.5);
    HybridSettings::setPointChargeThickness(59.0);
    EXPECT_THROW(setup.checkZoneRadii(), InputFileException);

    HybridSettings::setPointChargeThickness(2.0);
    EXPECT_NO_THROW(setup.checkZoneRadii());
}

TEST_F(TestSetup, hybridSetupRejectsMmChargesForMoltypeZero)
{
    _engine->getSimulationBox().addMoleculeType(molsys::MoleculeType(0));
    HybridSettings::setUseQMCharges(false);
    HybridSetup setup(*_engine);

    EXPECT_THROW(setup.validateQMChargeSettings(), InputFileException);

    HybridSettings::setUseQMCharges(true);
    EXPECT_NO_THROW(setup.validateQMChargeSettings());
}
