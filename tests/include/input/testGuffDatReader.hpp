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

#ifndef _TEST_GUFFDAT_READER_HPP_

#define _TEST_GUFFDAT_READER_HPP_

#include "atom.hpp"                      // for Atom
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "engine.hpp"                    // for Engine
#include "fileSettings.hpp"              // for FileSettings
#include "guffDatReader.hpp"             // for GuffDatReader
#include "guffNonCoulomb.hpp"            // for GuffNonCoulomb
#include "molecule.hpp"                  // for Molecule
#include "moleculeType.hpp"              // for MoleculeType
#include "potential.hpp"                 // for PotentialBruteForce, Potential
#include "potentialSettings.hpp"         // for PotentialSettings
#include "simulationBox.hpp"             // for SimulationBox

#include <gtest/gtest.h>   // for Test
#include <memory>          // for __shared_ptr_access, make_shared

/**
 * @class TestGuffDatReader
 *
 * @brief Fixture for guffDatReader tests.
 *
 */
class TestGuffDatReader : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        auto moleculeType1 = simulationBox::MoleculeType();
        moleculeType1.setNumberOfAtoms(2);
        moleculeType1.setMoltype(1);
        moleculeType1.addExternalAtomType(1);
        moleculeType1.addExternalAtomType(2);
        moleculeType1.addExternalToInternalAtomTypeElement(1, 0);
        moleculeType1.addExternalToInternalAtomTypeElement(2, 1);
        moleculeType1.addPartialCharge(0.5);
        moleculeType1.addPartialCharge(-0.25);
        moleculeType1.addAtomType(0);
        moleculeType1.addAtomType(1);

        auto moleculeType2 = simulationBox::MoleculeType();
        moleculeType2.setNumberOfAtoms(1);
        moleculeType2.setMoltype(2);
        moleculeType2.addExternalAtomType(3);
        moleculeType2.addExternalToInternalAtomTypeElement(3, 0);
        moleculeType2.addPartialCharge(0.25);
        moleculeType2.addAtomType(0);

        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(2);
        molecule1.setMoltype(1);

        auto atom1 = std::make_shared<simulationBox::Atom>();
        auto atom2 = std::make_shared<simulationBox::Atom>();

        atom1->setExternalAtomType(1);
        atom2->setExternalAtomType(2);
        atom1->setPartialCharge(0.5);
        atom2->setPartialCharge(-0.25);
        atom1->setAtomType(0);
        atom2->setAtomType(1);

        molecule1.addAtom(atom1);
        molecule1.addAtom(atom2);

        _engine = new engine::Engine();
        _engine->getSimulationBox().addMoleculeType(moleculeType1);
        _engine->getSimulationBox().addMoleculeType(moleculeType2);
        _engine->getSimulationBox().addMolecule(molecule1);

        settings::PotentialSettings::setCoulombRadiusCutOff(12.5);

        _engine->makePotential(potential::PotentialBruteForce());
        _engine->getPotential().makeNonCoulombPotential(potential::GuffNonCoulomb());
        _engine->getPotential().makeCoulombPotential(
            potential::CoulombShiftedPotential(settings::PotentialSettings::getCoulombRadiusCutOff()));

        settings::FileSettings::setGuffDatFileName("data/guffDatReader/guff.dat");
        settings::PotentialSettings::setNonCoulombType("guff");

        _guffDatReader = new input::guffdat::GuffDatReader(*_engine);
    }

    void TearDown() override
    {
        delete _guffDatReader;
        delete _engine;
    }

    input::guffdat::GuffDatReader *_guffDatReader;
    engine::Engine                *_engine;
};

#endif   // _TEST_GUFFDAT_READER_HPP_