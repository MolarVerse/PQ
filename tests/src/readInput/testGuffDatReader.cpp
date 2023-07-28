#include "testGuffDatReader.hpp"

#include "exceptions.hpp"

TEST_F(TestGuffDatReader, parseLineErrorMoltypeNotFound)
{
    auto line = std::vector<std::string>{"3", "1", "1"};
    EXPECT_THROW(_guffDatReader->parseLine(line), customException::GuffDatException);

    line = std::vector<std::string>{"1", "1", "3"};
    EXPECT_THROW(_guffDatReader->parseLine(line), customException::GuffDatException);
}

TEST_F(TestGuffDatReader, parseLineErrorAtomtypeNotFound)
{
    auto line = std::vector<std::string>{"1", "0", "2", "3"};
    EXPECT_THROW(_guffDatReader->parseLine(line), customException::GuffDatException);

    line = std::vector<std::string>{"1", "1", "2", "1"};
    EXPECT_THROW(_guffDatReader->parseLine(line), customException::GuffDatException);
}

TEST_F(TestGuffDatReader, setupGuffMaps)
{
    EXPECT_NO_THROW(_guffDatReader->setupGuffMaps());
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients().size(), 2);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients()[0].size(), 2);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients()[0].size(), 2);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients()[0][0].size(), 2);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients()[0][1].size(), 2);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients()[1][0].size(), 1);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients()[1][1].size(), 1);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients()[0][0][0].size(), 2);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients()[0][0][1].size(), 2);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients()[0][1][0].size(), 1);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients()[0][1][1].size(), 1);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients()[1][0][0].size(), 2);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients()[1][1][0].size(), 1);
}

TEST_F(TestGuffDatReader, parseLine)
{
    auto line = std::vector<std::string>{"1",   "1",   "2",   "3",   "-1.0", "10.0", "2.0", "2.0", "2.0", "2.0",
                                         "2.0", "2.0", "2.0", "2.0", "2.0",  "2.0",  "2.0", "2.0", "2.0", "2.0",
                                         "2.0", "2.0", "2.0", "2.0", "2.0",  "2.0",  "2.0", "2.0"};
    _guffDatReader->setupGuffMaps();
    EXPECT_NO_THROW(_guffDatReader->parseLine(line));

    EXPECT_EQ(_engine->getSimulationBox().getCoulombCoefficient(1, 2, 0, 0), 10.0);

    EXPECT_EQ(_engine->getSimulationBox().getRncCutOff(1, 2, 0, 0), 12.5);

    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[0], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[1], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[2], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[3], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[4], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[5], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[6], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[7], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[8], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[9], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[10], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[11], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[12], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[13], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[14], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[15], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[16], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[17], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[18], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[19], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[20], 2.0);
    EXPECT_EQ(_engine->getSimulationBox().getGuffCoefficients(1, 2, 0, 0)[21], 2.0);
}

TEST_F(TestGuffDatReader, readErrorNumberOfLineArguments)
{
    _guffDatReader->setFilename("data/guffDatReader/guffNumberLineElementsError.dat");
    EXPECT_THROW(_guffDatReader->read(), customException::GuffDatException);
}

/**
 * @brief readGuffdat function testing
 *
 * TODO: chdir has to be used because "guff.dat" is hardcoded in the class GuffDatReader
 *
 */
TEST_F(TestGuffDatReader, readGuffDat)
{
    chdir("data/guffDatReader");
    _guffDatReader->setFilename("data/guffDatReader/guff.dat");
    EXPECT_NO_THROW(readInput::readGuffDat(*_engine));
    chdir("../../");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}