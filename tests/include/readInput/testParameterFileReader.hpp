#ifndef _TEST_PARAMETER_FILE_READER_HPP_

#define _TEST_PARAMETER_FILE_READER_HPP_

#include "engine.hpp"
#include "parameterFileReader.hpp"

#include <gtest/gtest.h>
#include <string>

/**
 * @class TestParameterFileReader
 *
 * @brief Fixture class for testing the ParameterFileReader class
 *
 */
class TestParameterFileReader : public ::testing::Test
{
  protected:
    engine::Engine                                *_engine;
    readInput::parameterFile::ParameterFileReader *_parameterFileReader;

    void SetUp() override
    {
        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(1);

        auto molecule2 = simulationBox::Molecule();
        molecule2.setNumberOfAtoms(2);

        _engine = new engine::Engine();

        _engine->getSimulationBox().addMolecule(molecule1);
        _engine->getSimulationBox().addMolecule(molecule2);

        _parameterFileReader =
            new readInput::parameterFile::ParameterFileReader("data/parameterFileReader/param.param", *_engine);
    }

    void TearDown() override
    {
        delete _parameterFileReader;
        delete _engine;
    }
};

#endif   // _TEST_PARAMETER_FILE_READER_HPP_