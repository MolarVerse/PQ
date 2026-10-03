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

#include <gtest/gtest.h>

#include <memory>

#include "atom.hpp"
#include "coulombShiftedPotential.hpp"
#include "fileSettings.hpp"
#include "guffDatReader.hpp"
#include "guffNonCoulomb.hpp"
#include "mmmdEngine.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "potentialBruteForce.hpp"
#include "potentialSettings.hpp"

/**
 * @class TestGuffDatReader
 *
 * @brief Fixture for guffDatReader tests.
 *
 */
class TestGuffDatReader : public ::testing::Test
{
   protected:
    std::unique_ptr<input::guffdat::GuffDatReader> _guffDatReader;
    std::unique_ptr<engine::Engine>                _engine;

    void SetUp() override
    {
        auto moleculeType1 = molsys::MoleculeType();
        moleculeType1.setNumberOfAtoms(2);
        moleculeType1.setMoltype(MolType{1});
        moleculeType1.addExternalAtomType(ExtAtomType{1});
        moleculeType1.addExternalAtomType(ExtAtomType{2});
        moleculeType1.addExternalToInternalAtomTypeElement(
            ExtAtomType{1},
            AtomType{0}
        );
        moleculeType1.addExternalToInternalAtomTypeElement(
            ExtAtomType{2},
            AtomType{1}
        );
        moleculeType1.addPartialCharge(0.5);
        moleculeType1.addPartialCharge(-0.25);
        moleculeType1.addAtomType(AtomType{0});
        moleculeType1.addAtomType(AtomType{1});

        auto moleculeType2 = molsys::MoleculeType();
        moleculeType2.setNumberOfAtoms(1);
        moleculeType2.setMoltype(MolType{2});
        moleculeType2.addExternalAtomType(ExtAtomType{3});
        moleculeType2.addExternalToInternalAtomTypeElement(
            ExtAtomType{3},
            AtomType{0}
        );
        moleculeType2.addPartialCharge(0.25);
        moleculeType2.addAtomType(AtomType{0});

        auto molecule1 = molsys::Molecule();
        molecule1.setMoltype(MolType{1});

        auto atom1 = std::make_shared<molsys::Atom>();
        auto atom2 = std::make_shared<molsys::Atom>();

        atom1->setExternalAtomType(ExtAtomType{1});
        atom2->setExternalAtomType(ExtAtomType{2});
        atom1->setPartialCharge(0.5);
        atom2->setPartialCharge(-0.25);
        atom1->setAtomType(AtomType{0});
        atom2->setAtomType(AtomType{1});

        molecule1.addAtom(atom1);
        molecule1.addAtom(atom2);

        // NOTE: use dummy engine for testing
        //       this is implemented by base class Engine
        //       and works therefore for all derived classes
        _engine = std::make_unique<engine::MMMDEngine>();

        _engine->getSimulationBox().addMoleculeType(moleculeType1);
        _engine->getSimulationBox().addMoleculeType(moleculeType2);
        _engine->getSimulationBox().addMolecule(molecule1);

        settings::PotentialSettings::setCoulombRadiusCutOff(12.5);

        _engine->makePotential(pot::PotentialBruteForce());
        _engine->getPotential()->makeNonCoulombPotential(pot::GuffNonCoulomb());
        _engine->getPotential()->makeCoulombPotential(
            pot::CoulombShiftedPotential(
                settings::PotentialSettings::getCoulombRadiusCutOff()
            )
        );

        settings::FileSettings::setGuffDatFileName(
            "data/guffDatReader/guff.dat"
        );
        settings::PotentialSettings::setNonCoulombType(NonCoulombType::GUFF);

        _guffDatReader =
            std::make_unique<input::guffdat::GuffDatReader>(*_engine);
    }
};

#endif   // _TEST_GUFFDAT_READER_HPP_
