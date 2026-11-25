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
#include <vector>

#include "atom.hpp"
#include "enums/qm.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "hybridSetup.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "orthorhombicBox.hpp"
#include "qmSettings.hpp"
#include "testSetup.hpp"
#include "throwWithMessage.hpp"
#include "units.hpp"

namespace
{
    void addSingleAtomMolecule(engine::Engine &engine, MolType molType)
    {
        auto atom = std::make_shared<molsys::Atom>();
        atom->setPosition(
            linalg::Vec3D{static_cast<double>(molType.get()), 0.0, 0.0} *
            angstrom
        );

        molsys::Molecule molecule;
        molecule.setMoltype(molType);
        molecule.addAtom(atom);

        engine.get_simulationBox().addAtom(atom);
        engine.get_simulationBox().addMolecule(molecule);
    }

    void configureValidHybridSettings(
        Settings       &settings,
        engine::Engine &engine
    )
    {
        settings.generalSettings.setJobtypeByUser(JobType::QMMM_MD);
        settings.qmSettings.setQMMethodByUser(QMMethod::DFTBPLUS);
        settings.hybridSettings.set_forcedCoreListByUser({});
        settings.hybridSettings.set_forcedLayerListByUser({});
        settings.hybridSettings.set_forcedOuterListByUser({});
        settings.hybridSettings.set_useQMChargesByUser(true);
        settings.hybridSettings.set_coreRadiusByUser(2.0 * angstrom);
        settings.hybridSettings.set_layerRadiusByUser(4.0 * angstrom);
        settings.hybridSettings.set_smoothingRegionThicknessByUser(
            1.0 * angstrom
        );
        settings.hybridSettings.set_pointChargeThicknessByUser(2.0 * angstrom);

        molsys::OrthorhombicBox orthoBox{
            {40.0 * angstrom, 40.0 * angstrom, 40.0 * angstrom}
        };
        engine.get_simulationBox().setBox(orthoBox, false);
    }

}   // namespace

/* ---------- free function ---------- */

TEST_F(TestSetup, setupHybridIsNoOpWhenQMMMNotActive)
{
    // not QMMM_MD
    _settings->generalSettings.setJobtypeByUser(JobType::MM_MD);
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
        "Supported QM methods are \"dftbplus\", \"turbomole\" and \"orca\"."
    );
}

TEST_F(TestSetup, setupHybridConfiguresDefaultCenter)
{
    configureValidHybridSettings(*_settings, *_engine);
    addSingleAtomMolecule(*_engine, MolType{1});

    EXPECT_NO_THROW(setup::setupHybrid(*_engine));
    EXPECT_EQ(
        _settings->hybridSettings.get_innerRegionCenter(),
        std::vector<size_t>{0}
    );
}

TEST_F(TestSetup, setupHybridConfiguresExplicitLists)
{
    configureValidHybridSettings(*_settings, *_engine);
    _settings->qmSettings.setQMMethodByUser(QMMethod::TURBOMOLE);
    _settings->hybridSettings.set_innerRegionCenterByUser({0, 1});
    _settings->hybridSettings.set_forcedCoreListByUser({0});
    _settings->hybridSettings.set_forcedLayerListByUser({1});
    _settings->hybridSettings.set_forcedOuterListByUser({2});
    _settings->hybridSettings.set_useQMChargesByUser(false);
    addSingleAtomMolecule(*_engine, MolType{1});
    addSingleAtomMolecule(*_engine, MolType{2});
    addSingleAtomMolecule(*_engine, MolType{3});

    EXPECT_NO_THROW(setup::HybridSetup{*_engine}.setup());
    EXPECT_TRUE(_engine->get_simulationBox().getMolecule(0).isForcedCore());
    EXPECT_TRUE(_engine->get_simulationBox().getMolecule(1).isForcedLayer());
    EXPECT_TRUE(_engine->get_simulationBox().getMolecule(2).isForcedOuter());
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
                "calculations. Supported QM methods are \"dftbplus\", "
                "\"turbomole\" and \"orca\"."
        );
    }

    EXPECT_NO_THROW(setup.validateQMMethod(QMMethod::DFTBPLUS));
    EXPECT_NO_THROW(setup.validateQMMethod(QMMethod::TURBOMOLE));
    EXPECT_NO_THROW(setup.validateQMMethod(QMMethod::ORCA));
}

TEST_F(TestSetup, hybridSetupValidatesZoneRadii)
{
    molsys::OrthorhombicBox orthoBox{
        {40.0 * angstrom, 40.0 * angstrom, 40.0 * angstrom}
    };
    _engine->get_simulationBox().setBox(orthoBox, false);
    setup::HybridSetup setup{*_engine};

    _settings->hybridSettings.set_coreRadiusByUser(5.0 * angstrom);
    _settings->hybridSettings.set_layerRadiusByUser(4.0 * angstrom);
    _settings->hybridSettings.set_smoothingRegionThicknessByUser(
        1.0 * angstrom
    );
    _settings->hybridSettings.set_pointChargeThicknessByUser(0.0 * angstrom);
    EXPECT_THROW_MSG(
        setup.checkZoneRadii(),
        exc::InputFileException,
        "Core radius (5 Å) cannot be larger than layer radius (4 Å)"
    );

    _settings->hybridSettings.set_coreRadiusByUser(3.5 * angstrom);
    _settings->hybridSettings.set_layerRadiusByUser(4.0 * angstrom);
    _settings->hybridSettings.set_smoothingRegionThicknessByUser(
        1.0 * angstrom
    );
    EXPECT_THROW_MSG(
        setup.checkZoneRadii(),
        exc::InputFileException,
        "Smoothing region is too thick (1 Å) for the chosen combination of "
        "core (3.5 Å) and layer radius (4 Å)"
    );

    _settings->hybridSettings.set_coreRadiusByUser(2.0 * angstrom);
    _settings->hybridSettings.set_layerRadiusByUser(11.0 * angstrom);
    _settings->hybridSettings.set_smoothingRegionThicknessByUser(
        1.0 * angstrom
    );
    EXPECT_THROW_MSG(
        setup.checkZoneRadii(),
        exc::InputFileException,
        "Layer radius (11 Å) exceeds one quarter of the smallest box dimension "
        "(40 Å). This configuration is not allowed to ensure compliance with "
        "the minimum image convention."
    );

    _settings->hybridSettings.set_coreRadiusByUser(1.0 * angstrom);
    _settings->hybridSettings.set_layerRadiusByUser(2.0 * angstrom);
    _settings->hybridSettings.set_smoothingRegionThicknessByUser(
        0.5 * angstrom
    );
    _settings->hybridSettings.set_pointChargeThicknessByUser(59.0 * angstrom);
    EXPECT_THROW_MSG(
        setup.checkZoneRadii(),
        exc::InputFileException,
        "Layer radius (2 Å) plus point charge thickness (59 Å) exceeds three "
        "halves of the smallest box dimension (40 Å). This configuration is "
        "not allowed, as it would include point charges from beyond the "
        "immediate neighboring cells."
    );

    _settings->hybridSettings.set_pointChargeThicknessByUser(2.0 * angstrom);
    EXPECT_NO_THROW(setup.checkZoneRadii());
}

TEST_F(TestSetup, hybridSetupRejectsMmChargesForMoltypeZero)
{
    _engine->get_simulationBox().addMoleculeType(
        molsys::MoleculeType(MolType{0})
    );
    _settings->hybridSettings.set_useQMChargesByUser(false);
    setup::HybridSetup setup{*_engine};

    EXPECT_THROW_MSG(
        setup.validateQMChargeSettings(),
        exc::InputFileException,
        "Invalid configuration: MM charges requested (qm_charges = mm) in "
        "input file but atoms with moltype \"0\" are present in the system. "
        "Either set \"qm_charges = qm\" or ensure all atoms have anon-zero "
        "moltype."
    );

    _settings->hybridSettings.set_useQMChargesByUser(true);
    EXPECT_NO_THROW(setup.validateQMChargeSettings());
}

TEST_F(TestSetup, hybridSetupRejectsMoltypeZeroWithReferenceEnergies)
{
    auto &simBox = _engine->get_simulationBox();
    simBox.addMoleculeType(molsys::MoleculeType(MolType{0}));
    setup::HybridSetup setup{*_engine};

    EXPECT_NO_THROW(setup.validateQMReferenceEnergies());

    auto water = molsys::MoleculeType(MolType{1});
    water.setQMReferenceEnergy(-10.0 * kcal_per_mol);
    simBox.addMoleculeType(water);

    EXPECT_THROW_MSG(
        setup.validateQMReferenceEnergies(),
        exc::InputFileException,
        "Atoms with moltype \"0\" are not allowed when QM reference "
        "energies are set in the moldescriptor file."
    );
}
