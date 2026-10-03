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

#ifndef _TEST_BOND_CONSTRAINT_HPP_

#define _TEST_BOND_CONSTRAINT_HPP_

#include <gtest/gtest.h>

#include <memory>

#include "atom.hpp"
#include "bondConstraint.hpp"
#include "molecule.hpp"
#include "simulationBox.hpp"

/**
 * @class TestBondConstraint
 *
 * @brief Fixture for bond constraint tests.
 *
 */
class TestBondConstraint : public ::testing::Test
{
   protected:
    std::unique_ptr<molsys::SimulationBox>       _box;
    std::unique_ptr<constraints::BondConstraint> _bondConstraint;
    double                                       _targetBondLength = 1.2;

    void SetUp() override
    {
        auto molecule1 = molsys::Molecule();

        auto atom1 = std::make_shared<molsys::Atom>();
        auto atom2 = std::make_shared<molsys::Atom>();

        atom1->setPosition(linalg::Vec3D(1.0, 1.0, 1.0));
        atom2->setPosition(linalg::Vec3D(1.0, 2.0, 3.0));

        atom1->setMass(1.0);
        atom2->setMass(2.0);

        atom1->setVelocity(linalg::Vec3D(0.0, 0.0, 0.0));
        atom2->setVelocity(linalg::Vec3D(1.0, 1.0, 1.0));

        molecule1.addAtom(atom1);
        molecule1.addAtom(atom2);

        _box = std::make_unique<molsys::SimulationBox>();
        _box->addMolecule(molecule1);
        _box->setBoxDimensions(linalg::Vec3D(10.0, 10.0, 10.0));

        _bondConstraint = std::make_unique<constraints::BondConstraint>(
            _box->getMolecules().data(),
            _box->getMolecules().data(),
            AtomIndex{0},
            AtomIndex{1},
            _targetBondLength
        );
    }
};

#endif   // _TEST_BOND_CONSTRAINT_HPP_
