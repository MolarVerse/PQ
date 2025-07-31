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

#ifndef _TEST_THERMOSTAT_HPP_

#define _TEST_THERMOSTAT_HPP_

#include <gtest/gtest.h>   // for Test

#include <memory>   // for make_shared, __shared_ptr_access, shared_ptr

#include "atom.hpp"            // for Atom
#include "molecule.hpp"        // for Molecule
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox
#include "thermostat.hpp"      // for Thermostat
#include "vector3d.hpp"        // for Vec3D

class TestThermostat : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        _thermostat = new thermostat::Thermostat();
        _data       = new physicalData::PhysicalData();

        _simulationBox = new simulationBox::SimulationBox();

        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(2);

        auto atom1 = std::make_shared<simulationBox::Atom>();
        auto atom2 = std::make_shared<simulationBox::Atom>();

        atom1->setMass(1.0);
        atom2->setMass(1.0);
        atom1->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2->setVelocity(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        molecule1.setMolMass(2.0);
        molecule1.addAtom(atom1);
        molecule1.addAtom(atom2);

        auto molecule2 = simulationBox::Molecule();
        molecule2.setNumberOfAtoms(1);

        auto atom3 = std::make_shared<simulationBox::Atom>();

        atom3->setMass(1.0);
        atom3->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule2.setMolMass(1.0);

        molecule2.addAtom(atom3);

        _simulationBox->addMolecule(molecule1);
        _simulationBox->addMolecule(molecule2);
        _simulationBox->addAtom(atom1);
        _simulationBox->addAtom(atom2);
        _simulationBox->addAtom(atom3);

        _simulationBox->calculateDegreesOfFreedom();
    }

    void TearDown() override
    {
        delete _data;
        delete _simulationBox;
        delete _thermostat;
    }

    physicalData::PhysicalData   *_data;
    simulationBox::SimulationBox *_simulationBox;
    thermostat::Thermostat       *_thermostat;
};

#endif