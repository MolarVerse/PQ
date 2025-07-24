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

#include <gtest/gtest.h>   // for Test

#include <memory>   // for make_shared, __shared_ptr_access, shared_ptr
#include <vector>   // for vector

#include "atom.hpp"             // for Atom
#include "bondConstraint.hpp"   // for BondConstraint
#include "molecule.hpp"         // for Molecule
#include "simulationBox.hpp"    // for SimulationBox
#include "vector3d.hpp"         // for Vec3D

/**
 * @class TestBondConstraint
 *
 * @brief Fixture for bond constraint tests.
 *
 */
class TestBondConstraint : public ::testing::Test
{
   protected:
    virtual void SetUp()
    {
        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(3);

        auto atom1 = std::make_shared<simulationBox::Atom>();
        auto atom2 = std::make_shared<simulationBox::Atom>();

        atom1->setPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2->setPosition(linearAlgebra::Vec3D(1.0, 2.0, 3.0));

        atom1->setMass(1.0);
        atom2->setMass(2.0);

        atom1->setVelocity(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        atom2->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));

        molecule1.addAtom(atom1);
        molecule1.addAtom(atom2);

        _box = new simulationBox::SimulationBox();
        _box->addMolecule(molecule1);
        _box->setBoxDimensions(linearAlgebra::Vec3D(10.0, 10.0, 10.0));

        _bondConstraint = new constraints::BondConstraint(
            &(_box->getMolecules()[0]),
            &(_box->getMolecules()[0]),
            0,
            1,
            _targetBondLength
        );
    }

    virtual void TearDown()
    {
        delete _box;
        delete _bondConstraint;
    }

    simulationBox::SimulationBox *_box;
    constraints::BondConstraint  *_bondConstraint;
    double                        _targetBondLength = 1.2;
};

#endif   // _TEST_BOND_CONSTRAINT_HPP_