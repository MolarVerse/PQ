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

#include <gtest/gtest.h>   // for TEST, EXPECT_THROW_WITH_MESSAGE

#include <memory>          // for shared_ptr, make_shared
#include <unordered_set>   // for unordered_set

#include "atom.hpp"                 // for Atom
#include "exceptions.hpp"           // for HybridConfiguratorException
#include "hybridConfigurator.hpp"   // for HybridConfigurator
#include "hybridSettings.hpp"       // for HybridSettings
#include "molecule.hpp"             // for Molecule
#include "simulationBox.hpp"        // for SimulationBox
#include "throwWithMessage.hpp"     // for EXPECT_THROW_MSG
#include "vector3d.hpp"             // for Vec3D
#include "vectorNear.hpp"           // for EXPECT_VECTOR_NEAR

using namespace configurator;
using namespace customException;
using namespace linearAlgebra;
using namespace pq;
using namespace settings;
using namespace simulationBox;

TEST(testHybridConfigurator, calculateInnerRegionCenterAndShiftAtoms)
{
    HybridConfigurator hybridConfigurator;
    SimBox             simBox;

    EXPECT_THROW_MSG(
        hybridConfigurator.calculateInnerRegionCenter(simBox),
        HybridConfiguratorException,
        "Cannot calculate inner region center: no center atoms specified"
    );

    const auto center = Vec3D({0.43662598672853264, 0.60446640043437561, 0.0});

    auto atom1 = Atom();
    atom1.setPosition({0.40084395, 0.55383599, 0.0});
    atom1.setName("O");
    atom1.initMass();

    auto atom2 = Atom();
    atom2.setPosition({1.36084395, 0.55383599, 0.0});
    atom2.setName("H");
    atom2.initMass();

    auto atom3 = Atom();
    atom3.setPosition({0.08038937, 1.45877182, 0.0});
    atom3.setName("H");
    atom3.initMass();

    auto atom4 = Atom();
    atom4.setPosition({4.0, -4.5, -2.0});
    atom4.setName("Zr");
    atom4.initMass();

    simBox.addAtom(std::make_shared<Atom>(atom1));
    simBox.addAtom(std::make_shared<Atom>(atom2));
    simBox.addAtom(std::make_shared<Atom>(atom3));
    simBox.addAtom(std::make_shared<Atom>(atom4));

    simBox.setBoxDimensions({10.0, 10.0, 10.0});
    simBox.addInnerRegionCenterAtoms({0, 1, 2});

    hybridConfigurator.calculateInnerRegionCenter(simBox);

    EXPECT_VECTOR_NEAR(
        hybridConfigurator.getInnerRegionCenter(),
        center,
        1e-10
    );

    hybridConfigurator.shiftAtomsToInnerRegionCenter(simBox);

    EXPECT_VECTOR_NEAR(
        simBox.getAtom(0).getPosition(),
        Vec3D({-0.03578203672853264, -0.05063041043437561, 0.0}),
        1e-10
    );
    EXPECT_VECTOR_NEAR(
        simBox.getAtom(1).getPosition(),
        Vec3D({0.92421796327146732, -0.05063041043437561, 0.0}),
        1e-10
    );
    EXPECT_VECTOR_NEAR(
        simBox.getAtom(2).getPosition(),
        Vec3D({-0.35623661672853263, 0.85430541956562439, 0.0}),
        1e-10
    );
    EXPECT_VECTOR_NEAR(
        simBox.getAtom(3).getPosition(),
        Vec3D({3.5633740132714674, 4.8955335995656242, -2.0}),
        1e-10
    );

    hybridConfigurator.shiftAtomsBackToInitialPositions(simBox);

    EXPECT_VECTOR_NEAR(
        simBox.getAtom(0).getPosition(),
        atom1.getPosition(),
        1e-10
    );
    EXPECT_VECTOR_NEAR(
        simBox.getAtom(1).getPosition(),
        atom2.getPosition(),
        1e-10
    );
    EXPECT_VECTOR_NEAR(
        simBox.getAtom(2).getPosition(),
        atom3.getPosition(),
        1e-10
    );
    EXPECT_VECTOR_NEAR(
        simBox.getAtom(3).getPosition(),
        atom4.getPosition(),
        1e-10
    );
}

TEST(testHybridConfigurator, assignHybridZones)
{
    HybridConfigurator hybridConfigurator;
    SimBox             simBox;

    simBox.setBoxDimensions({100.0, 100.0, 100.0});

    HybridSettings::setCoreRadius(6.0);
    HybridSettings::setLayerRadius(12.0);
    HybridSettings::setSmoothingRegionThickness(2.0);
    HybridSettings::setPointChargeThickness(7.0);

    auto atom1 = std::make_shared<Atom>();
    atom1->setPosition({3.95, 3.15, 1.95}
    );   // C atom: r ≈ 5.41 < 6.0 (inside core)
    atom1->setName("C");
    atom1->initMass();

    auto atom2 = std::make_shared<Atom>();
    atom2->setPosition({4.45, 3.95, 2.45}
    );   // O atom: r ≈ 6.44 > 6.0 (outside core)
    atom2->setName("O");
    atom2->initMass();

    // mol1 com: (4.22, 3.58, 2.22), r ≈ 5.96 < 6.0 (inside core)
    auto mol1 = Molecule();
    mol1.addAtom(atom1);
    mol1.addAtom(atom2);
    mol1.setMolMass(atom1->getMass() + atom2->getMass());
    simBox.addMolecule(mol1);

    auto atom3 = std::make_shared<Atom>();
    atom3->setPosition({6.5, 5.0, 3.5});
    atom3->setName("Ar");
    atom3->initMass();

    auto mol2 = Molecule();
    mol2.addAtom(atom3);
    mol2.setMolMass(atom3->getMass());
    simBox.addMolecule(mol2);

    auto atom4 = std::make_shared<Atom>();
    atom4->setPosition({0.0, 0.0, 12.0});
    atom4->setName("Re");
    atom4->initMass();

    auto mol3 = Molecule();
    mol3.addAtom(atom4);
    mol3.setMolMass(atom4->getMass());
    simBox.addMolecule(mol3);

    auto atom5 = std::make_shared<Atom>();
    atom5->setPosition({0.0, 19.0, 0.0});
    atom5->setName("Zr");
    atom5->initMass();

    auto mol4 = Molecule();
    mol4.addAtom(atom5);
    mol4.setMolMass(atom5->getMass());
    simBox.addMolecule(mol4);

    auto atom6 = std::make_shared<Atom>();
    atom6->setPosition({19.0, 0.001, 0.0});
    atom6->setName("Tc");
    atom6->initMass();

    auto mol5 = Molecule();
    mol5.addAtom(atom6);
    mol5.setMolMass(atom6->getMass());
    simBox.addMolecule(mol5);

    hybridConfigurator.assignHybridZones(simBox);

    using enum simulationBox::HybridZone;
    EXPECT_EQ(simBox.getMolecule(0).getHybridZone(), CORE);
    EXPECT_EQ(simBox.getMolecule(1).getHybridZone(), LAYER);
    EXPECT_EQ(simBox.getMolecule(2).getHybridZone(), SMOOTHING);
    EXPECT_EQ(simBox.getMolecule(3).getHybridZone(), POINT_CHARGE);
    EXPECT_EQ(simBox.getMolecule(4).getHybridZone(), OUTER);

    EXPECT_EQ(hybridConfigurator.getNumberSmoothingMolecules(), 1);
}

TEST(testHybridConfigurator, activateDeactivateMolecules)
{
    HybridConfigurator hybridConfigurator;
    SimBox             simBox;

    using enum simulationBox::HybridZone;

    auto atom1 = std::make_shared<Atom>();
    auto mol1  = Molecule();
    mol1.addAtom(atom1);
    mol1.setHybridZone(CORE);

    auto atom2 = std::make_shared<Atom>();
    auto mol2  = Molecule();
    mol2.addAtom(atom2);
    mol2.setHybridZone(LAYER);

    auto atom3 = std::make_shared<Atom>();
    auto mol3  = Molecule();
    mol3.addAtom(atom3);
    mol3.setHybridZone(SMOOTHING);

    auto atom4 = std::make_shared<Atom>();
    auto mol4  = Molecule();
    mol4.addAtom(atom4);
    mol4.setHybridZone(SMOOTHING);

    auto atom5 = std::make_shared<Atom>();
    auto mol5  = Molecule();
    mol5.addAtom(atom5);
    mol5.setHybridZone(POINT_CHARGE);

    auto atom6 = std::make_shared<Atom>();
    auto mol6  = Molecule();
    mol6.addAtom(atom6);
    mol6.setHybridZone(OUTER);

    simBox.addMolecule(mol1);
    simBox.addMolecule(mol2);
    simBox.addMolecule(mol3);
    simBox.addMolecule(mol4);
    simBox.addMolecule(mol5);
    simBox.addMolecule(mol6);

    hybridConfigurator.activateMolecules(simBox);

    const auto &nMol = simBox.getMolecules().size();
    for (size_t i = 0; i < nMol; ++i)
    {
        EXPECT_EQ(simBox.getMolecule(i).isActive(), true);
        EXPECT_EQ(simBox.getMolecule(i).getAtom(0).isActive(), true);
    }

    hybridConfigurator.deactivateInnerMolecules(simBox);

    std::vector<bool> expected = {false, false, false, false, true, true};
    for (size_t i = 0; i < 6; ++i)
    {
        EXPECT_EQ(simBox.getMolecule(i).isActive(), expected[i]);
        EXPECT_EQ(simBox.getMolecule(i).getAtom(0).isActive(), expected[i]);
    }

    hybridConfigurator.activateMolecules(simBox);
    hybridConfigurator.deactivateOuterMolecules(simBox);

    expected = {true, true, true, true, false, false};
    for (size_t i = 0; i < 6; ++i)
    {
        EXPECT_EQ(simBox.getMolecule(i).isActive(), expected[i]);
        EXPECT_EQ(simBox.getMolecule(i).getAtom(0).isActive(), expected[i]);
    }

    hybridConfigurator.deactivateSmoothingMolecules(
        std::unordered_set<size_t>{0},
        simBox
    );

    EXPECT_EQ(simBox.getMolecule(2).isActive(), false);
    EXPECT_EQ(simBox.getMolecule(2).getAtom(0).isActive(), false);

    hybridConfigurator.activateSmoothingMolecules(
        std::unordered_set<size_t>{0},
        simBox
    );

    EXPECT_EQ(simBox.getMolecule(2).isActive(), true);
    EXPECT_EQ(simBox.getMolecule(2).getAtom(0).isActive(), true);
}
