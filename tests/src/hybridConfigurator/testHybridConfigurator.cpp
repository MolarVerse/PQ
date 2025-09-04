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

#include "atom.hpp"                 // for Atom
#include "exceptions.hpp"           // for HybridConfiguratorException
#include "hybridConfigurator.hpp"   // for HybridConfigurator
#include "simulationBox.hpp"        // for SimulationBox
#include "throwWithMessage.hpp"     // for EXPECT_THROW_MSG
#include "vector3d.hpp"             // for Vec3D
#include "vectorNear.hpp"           // for EXPECT_VECTOR_NEAR

using namespace pq;
using namespace customException;
using namespace configurator;
using namespace simulationBox;
using namespace linearAlgebra;

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