#ifndef _TEST_GUFFDAT_READER_HPP_

#define _TEST_GUFFDAT_READER_HPP_

#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "engine.hpp"                    // for Engine
#include "fileSettings.hpp"              // for FileSettings
#include "guffDatReader.hpp"             // for GuffDatReader
#include "guffNonCoulomb.hpp"            // for GuffNonCoulomb
#include "molecule.hpp"                  // for Molecule
#include "potential.hpp"                 // for PotentialBruteForce, Potential
#include "simulationBox.hpp"             // for SimulationBox

#include <gtest/gtest.h>   // for Test

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
        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(2);
        molecule1.setMoltype(1);
        molecule1.addExternalAtomType(1);
        molecule1.addExternalAtomType(2);
        molecule1.addExternalToInternalAtomTypeElement(1, 0);
        molecule1.addExternalToInternalAtomTypeElement(2, 1);
        molecule1.addPartialCharge(0.5);
        molecule1.addPartialCharge(-0.25);

        auto molecule2 = simulationBox::Molecule();
        molecule2.setNumberOfAtoms(1);
        molecule2.setMoltype(2);
        molecule2.addExternalAtomType(3);
        molecule2.addExternalToInternalAtomTypeElement(3, 0);

        _engine = new engine::Engine();
        _engine->getSimulationBox().addMoleculeType(molecule1);
        _engine->getSimulationBox().addMoleculeType(molecule2);
        _engine->getSimulationBox().addMolecule(molecule1);

        _engine->getSimulationBox().setCoulombRadiusCutOff(12.5);

        _engine->makePotential(potential::PotentialBruteForce());
        _engine->getPotential().makeNonCoulombPotential(potential::GuffNonCoulomb());
        _engine->getPotential().makeCoulombPotential(
            potential::CoulombShiftedPotential(_engine->getSimulationBox().getCoulombRadiusCutOff()));

        settings::FileSettings::setGuffDatFileName("data/guffDatReader/guff.dat");

        _guffDatReader = new readInput::guffdat::GuffDatReader(*_engine);
    }

    void TearDown() override
    {
        delete _guffDatReader;
        delete _engine;
    }

    readInput::guffdat::GuffDatReader *_guffDatReader;
    engine::Engine                    *_engine;
};

#endif   // _TEST_GUFFDAT_READER_HPP_