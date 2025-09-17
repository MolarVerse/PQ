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

#ifndef _TEST_VIRIAL_HPP_

#define _TEST_VIRIAL_HPP_

#include <gtest/gtest.h>   // for Test

#include <memory>   // for make_shared, __shared_ptr_access, shared_ptr

#include "atom.hpp"            // for Atom
#include "atomicVirial.hpp"    // for AtomicVirial
#include "molecule.hpp"        // for Molecule
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for Vec3D
#include "virial.hpp"          // for Virial

class TestVirial : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        _virial = new virial::AtomicVirial();
        _data   = new physicalData::PhysicalData();

        _simBox = new simulationBox::SimulationBox();

        auto molecule1 = simulationBox::Molecule();

        const auto atom1 = std::make_shared<simulationBox::Atom>();
        const auto atom2 = std::make_shared<simulationBox::Atom>();

        molecule1.setNumberOfAtoms(2);

        atom1->setPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2->setPosition(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        atom1->setForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2->setForce(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        atom1->setShiftForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2->setShiftForce(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        molecule1.setCenterOfMass(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule1.addAtom(atom1);
        molecule1.addAtom(atom2);

        auto molecule2 = simulationBox::Molecule();

        auto atom3 = std::make_shared<simulationBox::Atom>();

        molecule2.setNumberOfAtoms(1);
        atom3->setPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom3->setForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom3->setShiftForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule2.setCenterOfMass(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        molecule2.addAtom(atom3);

        _simBox->addMolecule(molecule1);
        _simBox->addMolecule(molecule2);

        _simBox->addAtom(atom1);
        _simBox->addAtom(atom2);
        _simBox->addAtom(atom3);

        _simBox->setBoxDimensions(linearAlgebra::Vec3D(10.0, 10.0, 10.0));
    }

    void TearDown() override
    {
        delete _data;
        delete _simBox;
        delete _virial;
    }

    physicalData::PhysicalData   *_data;
    simulationBox::SimulationBox *_simBox;
    virial::Virial               *_virial;
};

#endif