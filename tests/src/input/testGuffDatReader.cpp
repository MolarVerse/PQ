/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "testGuffDatReader.hpp"

#include "buckinghamPair.hpp"                        // for BuckinghamPair
#include "constants/internalConversionFactors.hpp"   // for _COULOMB_PREFACTOR_
#include "defaults.hpp"                              // for _NUMBER_OF_GUFF_ENTRIES_
#include "engine.hpp"                                // for Engine
#include "exceptions.hpp"                            // for GuffDatException, UserInputException
#include "forceFieldClass.hpp"                       // for ForceField
#include "guffPair.hpp"                              // for GuffPair
#include "lennardJonesPair.hpp"                      // for LennardJonesPair
#include "morsePair.hpp"                             // for MorsePair
#include "nonCoulombPair.hpp"                        // for NonCoulombPair
#include "nonCoulombPotential.hpp"                   // for NonCoulombPotential
#include "potentialSettings.hpp"                     // for PotentialSettings, string
#include "settings.hpp"                              // for Settings
#include "throwWithMessage.hpp"                      // for EXPECT_THROW_MSG

#include "gmock/gmock.h"   // for ElementsAre, MakePredicateFormatter
#include "gtest/gtest.h"   // for Message, TestPartResult
#include <cmath>           // for pow, exp
#include <cstddef>         // for size_t
#include <format>          // for format
#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)
#include <memory>          // for allocator, shared_ptr
#include <string>          // for string, basic_string, char_traits
#include <vector>          // for vector

using namespace input::guffdat;

/**
 * @brief tests parseLine function of GuffDatReader
 *
 * @details expects throw if mol types are not found
 *
 */
TEST_F(TestGuffDatReader, parseLine_ErrorMoltypeNotFound)
{
    auto line = std::vector<std::string>{"3", "1", "1"};
    EXPECT_THROW_MSG(_guffDatReader->parseLine(line), customException::GuffDatException, "Invalid molecule type in line 1");

    line = std::vector<std::string>{"1", "1", "3"};
    EXPECT_THROW_MSG(_guffDatReader->parseLine(line), customException::GuffDatException, "Invalid molecule type in line 1");
}

/**
 * @brief tests parseLine function of GuffDatReader
 *
 * @details expects throw if atom types are not found
 *
 */
TEST_F(TestGuffDatReader, parseLine_ErrorAtomTypeNotFound)
{
    auto line = std::vector<std::string>{"1", "0", "2", "3"};
    EXPECT_THROW_MSG(_guffDatReader->parseLine(line), customException::GuffDatException, "Invalid atom type in line 1");

    line = std::vector<std::string>{"1", "1", "2", "1"};
    EXPECT_THROW_MSG(_guffDatReader->parseLine(line), customException::GuffDatException, "Invalid atom type in line 1");
}

/**
 * @brief tests setupGuffMaps function of GuffDatReader
 *
 */
TEST_F(TestGuffDatReader, setupGuffMaps)
{
    EXPECT_NO_THROW(_guffDatReader->setupGuffMaps());

    const auto &potential = dynamic_cast<potential::GuffNonCoulomb &>(_engine->getPotential().getNonCoulombPotential());

    EXPECT_EQ(potential.getNonCoulombPairs().size(), 2);
    EXPECT_EQ(potential.getNonCoulombPairs()[0].size(), 2);
    EXPECT_EQ(potential.getNonCoulombPairs()[0].size(), 2);
    EXPECT_EQ(potential.getNonCoulombPairs()[0][0].size(), 2);
    EXPECT_EQ(potential.getNonCoulombPairs()[0][1].size(), 2);
    EXPECT_EQ(potential.getNonCoulombPairs()[1][0].size(), 1);
    EXPECT_EQ(potential.getNonCoulombPairs()[1][1].size(), 1);
    EXPECT_EQ(potential.getNonCoulombPairs()[0][0][0].size(), 2);
    EXPECT_EQ(potential.getNonCoulombPairs()[0][0][1].size(), 2);
    EXPECT_EQ(potential.getNonCoulombPairs()[0][1][0].size(), 1);
    EXPECT_EQ(potential.getNonCoulombPairs()[0][1][1].size(), 1);
    EXPECT_EQ(potential.getNonCoulombPairs()[1][0][0].size(), 2);
    EXPECT_EQ(potential.getNonCoulombPairs()[1][1][0].size(), 1);

    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients().size(), 2);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[0].size(), 2);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[1].size(), 2);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[0][0].size(), 2);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[0][1].size(), 2);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[1][0].size(), 1);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[1][1].size(), 1);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[0][0][0].size(), 2);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[0][0][1].size(), 2);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[0][1][0].size(), 1);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[0][1][1].size(), 1);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[1][0][0].size(), 2);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[1][1][0].size(), 1);

    EXPECT_EQ(_guffDatReader->getIsGuffPairSet().size(), 2);
    EXPECT_EQ(_guffDatReader->getIsGuffPairSet()[0].size(), 2);
    EXPECT_EQ(_guffDatReader->getIsGuffPairSet()[1].size(), 2);
    EXPECT_EQ(_guffDatReader->getIsGuffPairSet()[0][0].size(), 2);
    EXPECT_EQ(_guffDatReader->getIsGuffPairSet()[0][1].size(), 2);
    EXPECT_EQ(_guffDatReader->getIsGuffPairSet()[1][0].size(), 1);
    EXPECT_EQ(_guffDatReader->getIsGuffPairSet()[1][1].size(), 1);
    EXPECT_EQ(_guffDatReader->getIsGuffPairSet()[0][0][0].size(), 2);
    EXPECT_EQ(_guffDatReader->getIsGuffPairSet()[0][0][1].size(), 2);
    EXPECT_EQ(_guffDatReader->getIsGuffPairSet()[0][1][0].size(), 1);
    EXPECT_EQ(_guffDatReader->getIsGuffPairSet()[0][1][1].size(), 1);
    EXPECT_EQ(_guffDatReader->getIsGuffPairSet()[1][0][0].size(), 2);
    EXPECT_EQ(_guffDatReader->getIsGuffPairSet()[1][1][0].size(), 1);
}

/**
 * @brief tests parseLine function
 *
 */
TEST_F(TestGuffDatReader, parseLine)
{
    const auto line = std::vector<std::string>{"1",   "2",   "2",   "3",   "-1.0", "10.0", "2.0", "2.0", "3.0", "2.0",
                                               "2.0", "2.0", "2.0", "2.0", "2.0",  "2.0",  "2.0", "2.0", "2.0", "2.0",
                                               "2.0", "2.0", "2.0", "2.0", "2.0",  "2.0",  "2.0", "2.0"};
    _guffDatReader->setupGuffMaps();
    settings::PotentialSettings::setNonCoulombType("lj");
    EXPECT_NO_THROW(_guffDatReader->parseLine(line));

    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[0][1][1][0], 10.0);
    EXPECT_EQ(_guffDatReader->getGuffCoulombCoefficients()[1][0][0][1], 10.0);
    EXPECT_TRUE(_guffDatReader->getIsGuffPairSet()[0][1][1][0]);
    EXPECT_TRUE(_guffDatReader->getIsGuffPairSet()[1][0][0][1]);

    auto       &potential = dynamic_cast<potential::GuffNonCoulomb &>(_engine->getPotential().getNonCoulombPotential());
    const auto &pair      = dynamic_cast<potential::LennardJonesPair &>(*potential.getNonCoulombPair({1, 2, 1, 0}).get());

    EXPECT_EQ(pair, potential::LennardJonesPair(settings::PotentialSettings::getCoulombRadiusCutOff(), 2.0, 3.0));
}

TEST_F(TestGuffDatReader, addLennardJonesPair)
{
    _guffDatReader->setupGuffMaps();

    _guffDatReader->addLennardJonesPair(1, 2, 0, 0, {1.0, 2.0, 3.0}, 10.0);

    const auto &pair = dynamic_cast<potential::LennardJonesPair &>(
        *(_engine->getPotential().getNonCoulombPotential().getNonCoulombPair({1, 2, 0, 0}).get()));

    EXPECT_EQ(pair.getC6(), 1.0);
    EXPECT_EQ(pair.getC12(), 3.0);
    EXPECT_EQ(pair.getRadialCutOff(), 10.0);
    EXPECT_EQ(pair.getEnergyCutOff(), 1.0 / ::pow(10.0, 6) + 3.0 / ::pow(10.0, 12));
    EXPECT_EQ(pair.getForceCutOff(), 6.0 / ::pow(10.0, 7) + 12.0 * 3.0 / ::pow(10.0, 13));

    const auto &pair2 = dynamic_cast<potential::LennardJonesPair &>(
        *(_engine->getPotential().getNonCoulombPotential().getNonCoulombPair({2, 1, 0, 0}).get()));

    EXPECT_EQ(pair, pair2);
}

TEST_F(TestGuffDatReader, addBuckinghamPair)
{
    _guffDatReader->setupGuffMaps();

    _guffDatReader->addBuckinghamPair(1, 2, 0, 0, {1.0, 2.0, 3.0}, 10.0);

    const auto &pair = dynamic_cast<potential::BuckinghamPair &>(
        *(_engine->getPotential().getNonCoulombPotential().getNonCoulombPair({1, 2, 0, 0}).get()));

    EXPECT_EQ(pair.getA(), 1.0);
    EXPECT_EQ(pair.getDRho(), 2.0);
    EXPECT_EQ(pair.getC6(), 3.0);
    EXPECT_EQ(pair.getRadialCutOff(), 10.0);
    EXPECT_EQ(pair.getEnergyCutOff(), 1.0 * ::exp(10.0 * 2.0) + 3.0 / ::pow(10.0, 6));
    EXPECT_EQ(pair.getForceCutOff(), -2.0 * ::exp(10.0 * 2.0) + 6.0 * 3.0 / ::pow(10.0, 7));

    const auto &pair2 = dynamic_cast<potential::BuckinghamPair &>(
        *(_engine->getPotential().getNonCoulombPotential().getNonCoulombPair({2, 1, 0, 0}).get()));

    EXPECT_EQ(pair, pair2);
}

TEST_F(TestGuffDatReader, addMorsePair)
{
    _guffDatReader->setupGuffMaps();

    _guffDatReader->addMorsePair(1, 2, 0, 0, {1.0, 2.0, 3.0}, 10.0);

    const auto &pair = dynamic_cast<potential::MorsePair &>(
        *(_engine->getPotential().getNonCoulombPotential().getNonCoulombPair({1, 2, 0, 0}).get()));

    EXPECT_EQ(pair.getDissociationEnergy(), 1.0);
    EXPECT_EQ(pair.getWellWidth(), 2.0);
    EXPECT_EQ(pair.getEquilibriumDistance(), 3.0);
    EXPECT_EQ(pair.getRadialCutOff(), 10.0);
    EXPECT_EQ(pair.getEnergyCutOff(), 1.0 * ::pow((1.0 - ::exp(-2.0 * (10.0 - 3.0))), 2));
    EXPECT_EQ(pair.getForceCutOff(), -2.0 * (1.0 - ::exp(-2.0 * (10.0 - 3.0))) * ::exp(-2.0 * (10.0 - 3.0)) * 2.0);

    const auto &pair2 = dynamic_cast<potential::MorsePair &>(
        *(_engine->getPotential().getNonCoulombPotential().getNonCoulombPair({2, 1, 0, 0}).get()));

    EXPECT_EQ(pair, pair2);
}

TEST_F(TestGuffDatReader, addGuffPair)
{
    _guffDatReader->setupGuffMaps();

    _guffDatReader->addGuffPair(1, 2, 0, 0, {1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1}, 10.0);

    const auto &pair = dynamic_cast<potential::GuffPair &>(
        *(_engine->getPotential().getNonCoulombPotential().getNonCoulombPair({1, 2, 0, 0}).get()));

    EXPECT_THAT(pair.getCoefficients(),
                testing::ElementsAre(1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1));
    EXPECT_EQ(pair.getRadialCutOff(), 10.0);
    EXPECT_EQ(pair.getEnergyCutOff(), 3.0121946291700612e+35);
    EXPECT_EQ(pair.getForceCutOff(), -5.4219503325061099e+36);

    const auto &pair2 = dynamic_cast<potential::GuffPair &>(
        *(_engine->getPotential().getNonCoulombPotential().getNonCoulombPair({2, 1, 0, 0}).get()));

    EXPECT_THAT(pair2.getCoefficients(),
                testing::ElementsAre(1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1));
    EXPECT_EQ(pair2.getRadialCutOff(), 10.0);
    EXPECT_EQ(pair.getEnergyCutOff(), 3.0121946291700612e+35);
    EXPECT_EQ(pair.getForceCutOff(), -5.4219503325061099e+36);
}

TEST_F(TestGuffDatReader, addNonCoulombPair)
{
    const auto &guffCoefficients = std::vector<double>({1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1});
    _guffDatReader->setupGuffMaps();

    settings::PotentialSettings::setNonCoulombType("lj");
    _guffDatReader->addNonCoulombPair(1, 2, 0, 0, guffCoefficients, 10.0);

    EXPECT_NO_THROW([[maybe_unused]] const auto &dummy = dynamic_cast<potential::LennardJonesPair &>(
                        *(_engine->getPotential().getNonCoulombPotential().getNonCoulombPair({2, 1, 0, 0}).get())));

    settings::PotentialSettings::setNonCoulombType("buck");
    _guffDatReader->addNonCoulombPair(1, 2, 0, 0, guffCoefficients, 10.0);

    EXPECT_NO_THROW([[maybe_unused]] const auto &dummy = dynamic_cast<potential::BuckinghamPair &>(
                        *(_engine->getPotential().getNonCoulombPotential().getNonCoulombPair({2, 1, 0, 0}).get())));

    settings::PotentialSettings::setNonCoulombType("morse");
    _guffDatReader->addNonCoulombPair(1, 2, 0, 0, guffCoefficients, 10.0);

    EXPECT_NO_THROW([[maybe_unused]] const auto &dummy = dynamic_cast<potential::MorsePair &>(
                        *(_engine->getPotential().getNonCoulombPotential().getNonCoulombPair({2, 1, 0, 0}).get())));

    settings::PotentialSettings::setNonCoulombType("guff");
    _guffDatReader->addNonCoulombPair(1, 2, 0, 0, guffCoefficients, 10.0);

    EXPECT_NO_THROW([[maybe_unused]] const auto &dummy = dynamic_cast<potential::GuffPair &>(
                        *(_engine->getPotential().getNonCoulombPotential().getNonCoulombPair({2, 1, 0, 0}).get())));

    settings::PotentialSettings::setNonCoulombType("lj_9_12");

    EXPECT_THROW_MSG(
        _guffDatReader->addNonCoulombPair(1, 2, 0, 0, guffCoefficients, 10.0),
        customException::UserInputException,
        std::format("Invalid nonCoulombic type {} given", settings::string(settings::PotentialSettings::getNonCoulombType())));
}

/**
 * @brief tests read function
 *
 * @details error number of line arguments
 *
 */
TEST_F(TestGuffDatReader, read_errorNumberOfLineArguments)
{
    _guffDatReader->setFilename("data/guffDatReader/guffNumberLineElementsError.dat");
    EXPECT_THROW_MSG(_guffDatReader->read(),
                     customException::GuffDatException,
                     "Invalid number of commands (5) in line 3 - " + std::to_string(defaults::_NUMBER_OF_GUFF_ENTRIES_) +
                         " are allowed");
}

TEST_F(TestGuffDatReader, checkPartialCharges_NotMatchingCoefficients)
{
    _guffDatReader->setupGuffMaps();
    _guffDatReader->setGuffCoulombCoefficients(0, 0, 0, 0, 1.0);

    EXPECT_THROW_MSG(_guffDatReader->checkPartialCharges(),
                     customException::GuffDatException,
                     "Invalid coulomb coefficient guff file for molecule "
                     "types 1 and 1 and the 1. and the 1. atom type. The coulomb coefficient should "
                     "be 83.01592828541929 but is 1");
}

TEST_F(TestGuffDatReader, checkPartialCharges)
{
    _guffDatReader->setupGuffMaps();
    _guffDatReader->setGuffCoulombCoefficients(0, 0, 0, 0, constants::_COULOMB_PREFACTOR_ * 0.5 * 0.5);
    _guffDatReader->setGuffCoulombCoefficients(0, 0, 1, 0, -constants::_COULOMB_PREFACTOR_ * 0.5 * 0.25);
    _guffDatReader->setGuffCoulombCoefficients(0, 0, 0, 1, -constants::_COULOMB_PREFACTOR_ * 0.5 * 0.25);
    _guffDatReader->setGuffCoulombCoefficients(0, 0, 1, 1, constants::_COULOMB_PREFACTOR_ * 0.25 * 0.25);

    EXPECT_NO_THROW(_guffDatReader->checkPartialCharges());
}

TEST_F(TestGuffDatReader, checkNecessaryGuffPairs)
{
    engine::Engine              engine;
    simulationBox::Molecule     molecule1(1);
    simulationBox::Molecule     molecule2(2);
    simulationBox::MoleculeType moleculeType1(1);
    simulationBox::MoleculeType moleculeType2(2);
    simulationBox::MoleculeType moleculeType3(3);

    molecule1.setNumberOfAtoms(2);
    molecule2.setNumberOfAtoms(1);
    molecule2.setNumberOfAtoms(3);

    auto atom1 = std::make_shared<simulationBox::Atom>();
    auto atom2 = std::make_shared<simulationBox::Atom>();
    auto atom3 = std::make_shared<simulationBox::Atom>();

    atom1->setAtomType(0);
    atom2->setAtomType(1);
    atom1->setExternalAtomType(1);
    atom1->setExternalAtomType(4);
    molecule1.addAtom(atom1);
    molecule1.addAtom(atom2);

    moleculeType1.addAtomType(0);
    moleculeType1.addAtomType(1);
    moleculeType1.addExternalAtomType(1);
    moleculeType1.addExternalAtomType(4);
    moleculeType1.setNumberOfAtoms(2);

    atom3->setAtomType(0);
    atom3->setExternalAtomType(2);
    molecule2.addAtom(atom3);

    moleculeType2.addAtomType(0);
    moleculeType2.addExternalAtomType(2);
    moleculeType2.setNumberOfAtoms(1);

    moleculeType3.addAtomType(0);
    moleculeType3.addAtomType(1);
    moleculeType3.addAtomType(1);
    moleculeType3.addExternalAtomType(1);
    moleculeType3.addExternalAtomType(2);
    moleculeType3.addExternalAtomType(2);
    moleculeType3.setNumberOfAtoms(3);

    engine.getSimulationBox().addMolecule(molecule1);
    engine.getSimulationBox().addMolecule(molecule2);
    engine.getSimulationBox().addMoleculeType(moleculeType1);
    engine.getSimulationBox().addMoleculeType(moleculeType2);
    engine.getSimulationBox().addMoleculeType(moleculeType3);

    GuffDatReader guffDatReader(engine);

    engine.getPotential().setNonCoulombPotential(std::make_shared<potential::GuffNonCoulomb>());

    guffDatReader.setupGuffMaps();
    guffDatReader.setIsGuffPairSet(0, 0, 0, 0, true);
    guffDatReader.setIsGuffPairSet(0, 0, 1, 0, true);
    guffDatReader.setIsGuffPairSet(0, 0, 0, 1, true);
    guffDatReader.setIsGuffPairSet(0, 0, 1, 1, true);
    guffDatReader.setIsGuffPairSet(1, 0, 0, 0, true);
    guffDatReader.setIsGuffPairSet(1, 0, 0, 1, true);
    guffDatReader.setIsGuffPairSet(0, 1, 0, 0, true);
    guffDatReader.setIsGuffPairSet(0, 1, 1, 0, true);

    EXPECT_THROW_MSG(guffDatReader.checkNecessaryGuffPairs(),
                     customException::GuffDatException,
                     "No guff pair set for molecule types 2 and 2 and atom types 2 and the 2");

    guffDatReader.setIsGuffPairSet(1, 1, 0, 0, true);

    EXPECT_NO_THROW(guffDatReader.checkNecessaryGuffPairs());
}

TEST_F(TestGuffDatReader, calculatePartialCharges)
{
    _guffDatReader->setupGuffMaps();

    _guffDatReader->setGuffCoulombCoefficients(0, 0, 0, 0, constants::_COULOMB_PREFACTOR_ * 0.5 * 0.5);
    _guffDatReader->setGuffCoulombCoefficients(0, 0, 1, 1, constants::_COULOMB_PREFACTOR_ * 0.25 * 0.25);

    _guffDatReader->calculatePartialCharges();

    // EXPECT_EQ(_engine->getSimulationBox().getMolecule(0).getPartialCharges()[0], 0.5);
    // EXPECT_EQ(_engine->getSimulationBox().getMolecule(0).getPartialCharges()[1], -0.25);
}

/**
 * @brief readGuffdat function testing
 *
 */
TEST_F(TestGuffDatReader, readGuffDat)
{
    _guffDatReader->setFilename("data/guffDatReader/guff.dat");
    settings::Settings::activateMM();
    _engine->getForceFieldPtr()->activateNonCoulombic();
    settings::FileSettings::setGuffDatFileName("data/guffDatReader/guff.dat");
    EXPECT_NO_THROW(readGuffDat(*_engine));
}

TEST_F(TestGuffDatReader, readGuffDat_ErrorButNoThrowNotActivated)
{
    _guffDatReader->setFilename("");   // just to produce any kind of error
    settings::Settings::activateMM();
    _engine->getForceFieldPtr()->activateNonCoulombic();
    EXPECT_NO_THROW(input::guffdat::readGuffDat(*_engine));
}

TEST_F(TestGuffDatReader, readGuffDat_ErrorButNoThrowMMNotActivated)
{
    _guffDatReader->setFilename("");   // just to produce any kind of error
    settings::Settings::activateMM();
    _engine->getForceFieldPtr()->activateNonCoulombic();
    EXPECT_NO_THROW(input::guffdat::readGuffDat(*_engine));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}