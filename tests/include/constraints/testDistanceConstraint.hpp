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

#ifndef _TEST_DISTANCE_CONSTRAINT_HPP_

#define _TEST_DISTANCE_CONSTRAINT_HPP_

#include <gtest/gtest.h>

#include <memory>

#include "atom.hpp"
#include "distanceConstraint.hpp"
#include "molecule.hpp"
#include "simulationBox.hpp"

/**
 * @class TestDistanceConstraint
 *
 * @brief Fixture for distance constraint tests.
 *
 */
class TestDistanceConstraint : public ::testing::Test
{
   protected:
    std::unique_ptr<molsys::SimulationBox>           _box;
    std::unique_ptr<constraints::DistanceConstraint> _distanceConstraint;

    std::shared_ptr<molsys::Atom> _atom1;
    std::shared_ptr<molsys::Atom> _atom2;

    double _lowerDistance     = 1.0;
    double _upperDistance     = 3.0;
    double _springConstant    = 10.0;
    double _dSpringConstantDt = 0.0;

    void SetUp() override
    {
        auto molecule1 = molsys::Molecule();

        _atom1 = std::make_shared<molsys::Atom>();
        _atom2 = std::make_shared<molsys::Atom>();

        _atom1->setPosition(linalg::Vec3D(1.0, 1.0, 1.0));
        _atom2->setPosition(linalg::Vec3D(1.0, 2.0, 3.0));

        _atom1->setForceToZero();
        _atom2->setForceToZero();

        molecule1.addAtom(_atom1);
        molecule1.addAtom(_atom2);

        _box = std::make_unique<molsys::SimulationBox>();
        _box->addMolecule(molecule1);
        _box->setBoxDimensions(linalg::Vec3D(10.0, 10.0, 10.0));

        _distanceConstraint = std::make_unique<constraints::DistanceConstraint>(
            _box->getMolecules().data(),
            _box->getMolecules().data(),
            AtomIndex{0},
            AtomIndex{1},
            _lowerDistance,
            _upperDistance,
            _springConstant,
            _dSpringConstantDt
        );
    }
};

#endif   // _TEST_DISTANCE_CONSTRAINT_HPP_
