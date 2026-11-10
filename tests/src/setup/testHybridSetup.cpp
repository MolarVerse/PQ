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
#include "enums/qm.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "hybridSetup.hpp"
#include "inputFileParser/hybridInputParser.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "orthorhombicBox.hpp"
#include "qmSettings.hpp"
#include "testSetup.hpp"
#include "throwWithMessage.hpp"

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

    void configureValidHybridSettings(
        Settings       &settings,
        engine::Engine &engine
    )
    {
        settings::GeneralSettings::setJobtype(JobType::QMMM_MD);
        settings.qmSettings.setQMMethodByUser(QMMethod::DFTBPLUS);
        settings::HybridSettings::setForcedCoreList({});
        settings::HybridSettings::setForcedLayerList({});
        settings::HybridSettings::setForcedOuterList({});
        settings::HybridSettings::setUseQMCharges(true);
        settings::HybridSettings::setCoreRadius(2.0);
        settings::HybridSettings::setLayerRadius(4.0);
        settings::HybridSettings::setSmoothingRegionThickness(1.0);
        settings::HybridSettings::setPointChargeThickness(2.0);

        molsys::OrthorhombicBox orthoBox{{40.0, 40.0, 40.0}};
        engine.getSimulationBox().setBox(orthoBox, false);
    }

}   // namespace

/* ---------- free function ---------- */

TEST_F(TestSetup, setupHybridIsNoOpWhenQMMMNotActive)
{
    settings::GeneralSettings::setJobtype(JobType::MM_MD);   // not QMMM_MD
    EXPECT_NO_THROW(setup::setupHybrid(*_engine));
}

/* ---------- setup throws ---------- */

TEST_F(TestSetup, setupThrowsNotImplemented)
{
    setup::HybridSetup hybridSetup{*_engine};
    EXPECT_THROW_MSG(
        hybridSetup.setup(),
        exc::InputFileException,
        "QM method \"NONE\" is not supported for hybrid type calculations. "
        "Supported QM methods are \"dftbplus\" and \"turbomole\"."
    );
}

TEST_F(TestSetup, setupHybridConfiguresDefaultCenter)
{
    configureValidHybridSettings(*_settings, *_engine);
    addSingleAtomMolecule(*_engine, MolType{1});

    EXPECT_NO_THROW(setup::setupHybrid(*_engine));
    EXPECT_EQ(
        _engine->getSimulationBox().getInnerRegionCenterAtomIndices(),
        std::vector<size_t>{0}
    );
}

TEST_F(TestSetup, setupHybridConfiguresExplicitLists)
{
    configureValidHybridSettings(*_settings, *_engine);
    _settings->qmSettings.setQMMethodByUser(QMMethod::TURBOMOLE);
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
        QMMethod::PYSCF,
        QMMethod::ASE_DFTBPLUS,
        QMMethod::ASE_XTB,
        QMMethod::MACE,
        QMMethod::FENNOL,
        QMMethod::NONE,
    };

    for (const auto method : unsupported)
    {
        EXPECT_THROW_MSG(
            setup.validateQMMethod(method),
            exc::InputFileException,
            "QM method \"" + QMMethodMeta::toString(method) +
                "\" is not supported for hybrid type "
                "calculations. Supported QM methods are \"dftbplus\" and "
                "\"turbomole\"."
        );
    }

    EXPECT_NO_THROW(setup.validateQMMethod(QMMethod::DFTBPLUS));
    EXPECT_NO_THROW(setup.validateQMMethod(QMMethod::TURBOMOLE));
}

TEST_F(TestSetup, hybridSetupValidatesZoneRadii)
{
    molsys::OrthorhombicBox orthoBox{{40.0, 40.0, 40.0}};
    _engine->getSimulationBox().setBox(orthoBox, false);
    setup::HybridSetup setup{*_engine};

    settings::HybridSettings::setCoreRadius(5.0);
    settings::HybridSettings::setLayerRadius(4.0);
    settings::HybridSettings::setSmoothingRegionThickness(1.0);
    settings::HybridSettings::setPointChargeThickness(0.0);
    EXPECT_THROW_MSG(
        setup.checkZoneRadii(),
        exc::InputFileException,
        "Core radius (5 Å) cannot be larger than layer radius (4 Å)"
    );

    settings::HybridSettings::setCoreRadius(3.5);
    settings::HybridSettings::setLayerRadius(4.0);
    settings::HybridSettings::setSmoothingRegionThickness(1.0);
    EXPECT_THROW_MSG(
        setup.checkZoneRadii(),
        exc::InputFileException,
        "Smoothing region is too thick (1 Å) for the chosen combination of "
        "core (3.5 Å) and layer radius (4 Å)"
    );

    settings::HybridSettings::setCoreRadius(2.0);
    settings::HybridSettings::setLayerRadius(11.0);
    settings::HybridSettings::setSmoothingRegionThickness(1.0);
    EXPECT_THROW_MSG(
        setup.checkZoneRadii(),
        exc::InputFileException,
        "Layer radius (11 Å) exceeds one quarter of the smallest box dimension "
        "(40 Å). This configuration is not allowed to ensure compliance with "
        "the minimum image convention."
    );

    settings::HybridSettings::setCoreRadius(1.0);
    settings::HybridSettings::setLayerRadius(2.0);
    settings::HybridSettings::setSmoothingRegionThickness(0.5);
    settings::HybridSettings::setPointChargeThickness(59.0);
    EXPECT_THROW_MSG(
        setup.checkZoneRadii(),
        exc::InputFileException,
        "Layer radius (2 Å) plus point charge thickness (59 Å) exceeds three "
        "halves of the smallest box dimension (40 Å). This configuration is "
        "not allowed, as it would include point charges from beyond the "
        "immediate neighboring cells."
    );

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

    EXPECT_THROW_MSG(
        setup.validateQMChargeSettings(),
        exc::InputFileException,
        "Invalid configuration: MM charges requested (qm_charges = mm) in "
        "input file but atoms with moltype \"0\" are present in the system. "
        "Either set \"qm_charges = qm\" or ensure all atoms have anon-zero "
        "moltype."
    );

    settings::HybridSettings::setUseQMCharges(true);
    EXPECT_NO_THROW(setup.validateQMChargeSettings());
}

TEST_F(TestSetup, hybridSetupRejectsMoltypeZeroWithReferenceEnergies)
{
    auto &simBox = _engine->getSimulationBox();
    simBox.addMoleculeType(molsys::MoleculeType(MolType{0}));
    setup::HybridSetup setup{*_engine};

    EXPECT_NO_THROW(setup.validateQMReferenceEnergies());

    auto water = molsys::MoleculeType(MolType{1});
    water.setQMReferenceEnergy(-10.0);
    simBox.addMoleculeType(water);

    EXPECT_THROW_MSG(
        setup.validateQMReferenceEnergies(),
        exc::InputFileException,
        "Atoms with moltype \"0\" are not allowed when QM reference "
        "energies are set in the moldescriptor file."
    );
}
