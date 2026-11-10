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

#include <memory>
#include <vector>

#include "atom.hpp"
#include "frame.hpp"
#include "molecule.hpp"
#include "noseHooverThermostat.hpp"
#include "orthorhombicBox.hpp"
#include "simulationBox.hpp"
#include "staticMatrix.hpp"
#include "strongTypes.hpp"
#include "thermostat.hpp"
#include "thermostatSettings.hpp"
#include "unit_types.hpp"
#include "vector3d.hpp"

namespace
{
    std::shared_ptr<molsys::Atom> makeAtom(
        const char          *name,
        const linalg::Vec3D &position,
        double               charge
    )
    {
        auto atom = std::make_shared<molsys::Atom>();
        atom->setName(name);
        atom->setPosition(position);
        atom->setVelocity(2.0 * position);
        atom->setForce(-position);
        atom->setPartialCharge(charge);
        return atom;
    }

    molsys::SimulationBox makeSimulationBox()
    {
        molsys::SimulationBox simulationBox;
        simulationBox.setBox(
            molsys::OrthorhombicBox{{10.0, 10.0, 10.0}},
            false
        );

        const auto hydrogen = makeAtom("H", {1.0, 2.0, 3.0}, 0.4);
        const auto oxygen   = makeAtom("O", {4.0, 5.0, 6.0}, -0.8);
        const auto argon    = makeAtom("Ar", {7.0, 8.0, 9.0}, 0.0);

        molsys::Molecule water{MolType{1}};
        water.addAtom(hydrogen);
        water.addAtom(oxygen);

        molsys::Molecule noble{MolType{2}};
        noble.addAtom(argon);

        simulationBox.addMolecule(water);
        simulationBox.addMolecule(noble);
        simulationBox.addAtom(hydrogen);
        simulationBox.addAtom(oxygen);
        simulationBox.addAtom(argon);

        return simulationBox;
    }
}   // namespace

TEST(TestFrame, copiesAtomsInMoleculeOrder)
{
    const auto       simulationBox = makeSimulationBox();
    const out::Frame frame{simulationBox};

    ASSERT_EQ(frame.atoms.size(), 3);

    const auto &oxygen = frame.atoms[1];
    EXPECT_EQ(oxygen.name, "O");
    EXPECT_EQ(oxygen.moltype, MolType{1});
    EXPECT_EQ(oxygen.indexInMolecule, 1);
    EXPECT_EQ(oxygen.position, linalg::Vec3D(4.0, 5.0, 6.0));
    EXPECT_EQ(oxygen.velocity, linalg::Vec3D(8.0, 10.0, 12.0));
    EXPECT_EQ(oxygen.force, linalg::Vec3D(-4.0, -5.0, -6.0));
    EXPECT_DOUBLE_EQ(oxygen.charge, -0.8);

    const auto &argon = frame.atoms[2];
    EXPECT_EQ(argon.moltype, MolType{2});
    EXPECT_EQ(argon.indexInMolecule, 0);

    EXPECT_EQ(frame.box->get_boxDimensions(), linalg::Vec3D(10.0, 10.0, 10.0));
    EXPECT_DOUBLE_EQ(frame.totalForce(), simulationBox.calculateTotalForce());
}

TEST(TestFrame, qmChargeReplacesPartialCharge)
{
    auto simulationBox = makeSimulationBox();
    simulationBox.getAtoms()[1]->setQMCharge(-0.5);

    const out::Frame frame{simulationBox};

    EXPECT_DOUBLE_EQ(frame.atoms[1].charge, -0.5);
}

TEST(TestFrame, doesNotFollowLaterChanges)
{
    auto             simulationBox = makeSimulationBox();
    const out::Frame frame{simulationBox};

    simulationBox.getAtoms()[0]->setPosition({0.0, 0.0, 0.0});
    simulationBox.getBox().scaleBox(linalg::diagonalMatrix(2.0));

    EXPECT_EQ(frame.atoms[0].position, linalg::Vec3D(1.0, 2.0, 3.0));
    EXPECT_EQ(
        frame.box->wrapPositionIntoBox({6.0, 0.0, 0.0}),
        linalg::Vec3D(-4.0, 0.0, 0.0)
    );
}

TEST(TestFrame, copiesNoseHooverChain)
{
    const auto simulationBox = makeSimulationBox();

    NoseHooverSettings noseHooverSettings;
    noseHooverSettings.setChiByUser({{0, 1.0}, {1, 2.0}});
    noseHooverSettings.setZetaByUser({{0, 3.0}, {1, 4.0}});
    noseHooverSettings.setNoseHooverChainLengthByUser(2);

    const thermostat::NoseHooverThermostat noseHoover{
        noseHooverSettings,
        toTemperature(300.0)
    };

    const out::Frame withChain{simulationBox, noseHoover};
    const out::Frame withoutChain{simulationBox, thermostat::Thermostat()};

    const std::vector<out::Frame::ChainLink> expected{{1.0, 3.0}, {2.0, 4.0}};
    EXPECT_EQ(withChain.noseHooverChain, expected);
    EXPECT_TRUE(withoutChain.noseHooverChain.empty());
}
