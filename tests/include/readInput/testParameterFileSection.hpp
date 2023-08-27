#ifndef _TEST_PARAMETER_FILE_SECTION_HPP_

#define _TEST_PARAMETER_FILE_SECTION_HPP_

#include "engine.hpp"                 // for Engine
#include "forceFieldNonCoulomb.hpp"   // for ForceFieldNonCoulomb
#include "molecule.hpp"               // for Molecule
#include "potential.hpp"              // for Potential
#include "simulationBox.hpp"          // for SimulationBox

#include <gtest/gtest.h>   // for Test
#include <stdio.h>         // for remove
#include <string>          // for allocator, string

/**
 * @class TestParameterFileSection
 *
 * @brief Fixture class for testing the ParameterFileSection class
 *
 */
class TestParameterFileSection : public ::testing::Test
{
  protected:
    engine::Engine *_engine;
    std::string     _parameterFilename = "param.param";

    void SetUp() override
    {
        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(1);

        auto molecule2 = simulationBox::Molecule();
        molecule2.setNumberOfAtoms(3);

        _engine = new engine::Engine();

        _engine->getSimulationBox().addMolecule(molecule1);
        _engine->getSimulationBox().addMolecule(molecule2);

        _engine->getPotential().makeNonCoulombPotential(potential::ForceFieldNonCoulomb());
    }

    void TearDown() override
    {
        delete _engine;
        ::remove(_parameterFilename.c_str());
    }
};

#endif   // _TEST_PARAMETER_FILE_SECTION_HPP_