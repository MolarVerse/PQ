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

#ifndef TESTS__INCLUDE__PHYSICAL_DATA__TEST_PHYSICAL_DATA_HPP
#define TESTS__INCLUDE__PHYSICAL_DATA__TEST_PHYSICAL_DATA_HPP

#include <gtest/gtest.h>

#include <memory>

#include "atom.hpp"
#include "molecule.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "unit_types.hpp"
#include "units.hpp"

/**
 * @class TestPhysicalData
 *
 * @brief test suite for physical data
 *
 */
class TestPhysicalData : public ::testing::Test
{
   protected:
    std::unique_ptr<physicalData::PhysicalData> _physicalData;
    std::unique_ptr<molsys::SimulationBox>      _simulationBox;

    void SetUp() override
    {
        _physicalData = std::make_unique<physicalData::PhysicalData>();
        _physicalData->set_coulombEnergy(1.0 * kcal_per_mol);
        _physicalData->set_nonCoulombEnergy(2.0 * kcal_per_mol);
        _physicalData->set_temperature(toTemperature(3.0));
        _physicalData->set_momentum(
            linalg::Vec3D(4.0) * g_per_mol * angstrom / s
        );
        _physicalData->set_kineticEnergy(5.0 * kcal_per_mol);
        _physicalData->set_volume(6.0 * angstrom3);
        _physicalData->set_density(7.0 * kg_per_litre);
        _physicalData->set_pressure(8.0 * bar);
        _physicalData->set_coupledPressure(8.5 * bar);
        _physicalData->set_qmEnergy(9.0 * kcal_per_mol);
        _physicalData->set_qmReferenceEnergy(1.0 * kcal_per_mol);

        _simulationBox = std::make_unique<molsys::SimulationBox>();

        auto molecule1 = molsys::Molecule();

        auto atom1 = std::make_shared<molsys::Atom>();
        auto atom2 = std::make_shared<molsys::Atom>();

        atom1->setMass(1.0 * g_per_mol);
        atom2->setMass(1.0 * g_per_mol);
        atom1->setVelocity(linalg::Vec3D(1.0, 1.0, 1.0) * angstrom / s);
        atom2->setVelocity(linalg::Vec3D(1.0, 2.0, 3.0) * angstrom / s);
        molecule1.addAtom(atom1);
        molecule1.addAtom(atom2);

        auto molecule2 = molsys::Molecule();

        auto atom3 = std::make_shared<molsys::Atom>();
        atom3->setMass(1.0 * g_per_mol);
        atom3->setVelocity(linalg::Vec3D(1.0, 1.0, 1.0) * angstrom / s);
        molecule2.addAtom(atom3);

        _simulationBox->addMolecule(molecule1);
        _simulationBox->addMolecule(molecule2);

        _simulationBox->calculateDegreesOfFreedom();
    }
};

#endif
