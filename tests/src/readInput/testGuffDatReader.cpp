#include "testGuffDatReader.hpp"

#include "buckinghamPair.hpp"
#include "defaults.hpp"
#include "engine.hpp"
#include "exceptions.hpp"
#include "guffPair.hpp"
#include "lennardJonesPair.hpp"
#include "morsePair.hpp"
#include "potentialSettings.hpp"
#include "throwWithMessage.hpp"

#include <format>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace readInput::guffdat;

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

// // /**
// //  * @brief tests parseLine function
// //  *
// //  */
// // TEST_F(TestGuffDatReader, parseLine)
// // {
// //     auto line = std::vector<std::string>{"1",   "1",   "2",   "3",   "-1.0", "10.0", "2.0", "2.0", "2.0", "2.0",
// //                                          "2.0", "2.0", "2.0", "2.0", "2.0",  "2.0",  "2.0", "2.0", "2.0", "2.0",
// //                                          "2.0", "2.0", "2.0", "2.0", "2.0",  "2.0",  "2.0", "2.0"};
// //     _guffDatReader->setupGuffMaps();
// //     EXPECT_NO_THROW(_guffDatReader->parseLine(line));

// //     EXPECT_EQ(_engine->getSimulationBox().getCoulombCoefficient(1, 2, 0, 0), 10.0);

// //     EXPECT_EQ(_engine->getSimulationBox().getNonCoulombRadiusCutOff(1, 2, 0, 0), 12.5);

// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[0], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[1], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[2], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[3], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[4], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[5], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[6], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[7], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[8], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[9], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[10], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[11], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[12], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[13], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[14], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[15], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[16], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[17], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[18], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[19], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[20], 2.0);
// //     EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[21], 2.0);
// // }

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

// /**
//  * @brief readGuffdat function testing
//  *
//  */
// TEST_F(TestGuffDatReader, readGuffDat)
// {
//     _guffDatReader->setFilename("data/guffDatReader/guff.dat");
//     EXPECT_NO_THROW(readInput::readGuffDat(*_engine));
// }

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}