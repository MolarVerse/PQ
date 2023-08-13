#include "exceptions.hpp"
#include "parameterFileSection.hpp"
#include "testParameterFileSection.hpp"
#include "throwWithMessage.hpp"

using namespace ::testing;
using namespace readInput::parameterFile;

TEST_F(TestParameterFileSection, processSectionLennardJones)
{
    std::vector<std::string> lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    NonCoulombicsSection     nonCoulombicsSection;
    nonCoulombicsSection.processLJ(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getNonCoulombicPairsVector().size(), 1);
    auto *pair =
        dynamic_cast<const forceField::LennardJonesPair *>(_engine->getForceField().getNonCoulombicPairsVector()[0].get());
    EXPECT_EQ(pair->getVanDerWaalsType1(), 0);
    EXPECT_EQ(pair->getVanDerWaalsType2(), 1);
    EXPECT_EQ(pair->getC6(), 1.22);
    EXPECT_EQ(pair->getC12(), 234.3);
    EXPECT_EQ(pair->getRadialCutOff(), 324.3);

    lineElements = {"0", "1", "1.22", "234.3"};
    nonCoulombicsSection.processLJ(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getNonCoulombicPairsVector().size(), 2);
    auto *pair2 =
        dynamic_cast<const forceField::LennardJonesPair *>(_engine->getForceField().getNonCoulombicPairsVector()[1].get());
    EXPECT_EQ(pair2->getVanDerWaalsType1(), 0);
    EXPECT_EQ(pair2->getVanDerWaalsType2(), 1);
    EXPECT_EQ(pair2->getC6(), 1.22);
    EXPECT_EQ(pair2->getC12(), 234.3);
    EXPECT_EQ(pair2->getRadialCutOff(), -1.0);

    lineElements = {"1", "2", "1.0", "0", "2", "3.3"};
    EXPECT_THROW(nonCoulombicsSection.processLJ(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, processSectionLennardBuckingham)
{
    std::vector<std::string> lineElements = {"0", "1", "1.22", "234.3", "324.3", "435"};
    NonCoulombicsSection     nonCoulombicsSection;
    nonCoulombicsSection.processBuckingham(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getNonCoulombicPairsVector().size(), 1);
    auto *pair = dynamic_cast<const forceField::BuckinghamPair *>(_engine->getForceField().getNonCoulombicPairsVector()[0].get());
    EXPECT_EQ(pair->getVanDerWaalsType1(), 0);
    EXPECT_EQ(pair->getVanDerWaalsType2(), 1);
    EXPECT_EQ(pair->getA(), 1.22);
    EXPECT_EQ(pair->getDRho(), 234.3);
    EXPECT_EQ(pair->getC6(), 324.3);
    EXPECT_EQ(pair->getRadialCutOff(), 435.0);

    lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    nonCoulombicsSection.processBuckingham(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getNonCoulombicPairsVector().size(), 2);
    auto *pair2 =
        dynamic_cast<const forceField::BuckinghamPair *>(_engine->getForceField().getNonCoulombicPairsVector()[1].get());
    EXPECT_EQ(pair2->getVanDerWaalsType1(), 0);
    EXPECT_EQ(pair2->getVanDerWaalsType2(), 1);
    EXPECT_EQ(pair2->getA(), 1.22);
    EXPECT_EQ(pair2->getDRho(), 234.3);
    EXPECT_EQ(pair2->getC6(), 324.3);
    EXPECT_EQ(pair2->getRadialCutOff(), -1.0);

    lineElements = {"1", "2", "1.0", "0", "2", "3.3", "345"};
    EXPECT_THROW(nonCoulombicsSection.processBuckingham(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, processSectionLennardMorse)
{
    std::vector<std::string> lineElements = {"0", "1", "1.22", "234.3", "324.3", "435"};
    NonCoulombicsSection     nonCoulombicsSection;
    nonCoulombicsSection.processMorse(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getNonCoulombicPairsVector().size(), 1);
    auto *pair = dynamic_cast<const forceField::MorsePair *>(_engine->getForceField().getNonCoulombicPairsVector()[0].get());
    EXPECT_EQ(pair->getVanDerWaalsType1(), 0);
    EXPECT_EQ(pair->getVanDerWaalsType2(), 1);
    EXPECT_EQ(pair->getDissociationEnergy(), 1.22);
    EXPECT_EQ(pair->getWellWidth(), 234.3);
    EXPECT_EQ(pair->getEquilibriumDistance(), 324.3);
    EXPECT_EQ(pair->getRadialCutOff(), 435.0);

    lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    nonCoulombicsSection.processMorse(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getNonCoulombicPairsVector().size(), 2);
    auto *pair2 = dynamic_cast<const forceField::MorsePair *>(_engine->getForceField().getNonCoulombicPairsVector()[1].get());
    EXPECT_EQ(pair2->getVanDerWaalsType1(), 0);
    EXPECT_EQ(pair2->getVanDerWaalsType2(), 1);
    EXPECT_EQ(pair2->getDissociationEnergy(), 1.22);
    EXPECT_EQ(pair2->getWellWidth(), 234.3);
    EXPECT_EQ(pair2->getEquilibriumDistance(), 324.3);
    EXPECT_EQ(pair2->getRadialCutOff(), -1.0);

    lineElements = {"1", "2", "1.0", "0", "2", "3.3", "345"};
    EXPECT_THROW(nonCoulombicsSection.processMorse(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, processHeader)
{
    std::vector<std::string> lineElements = {"noncoulombics"};
    NonCoulombicsSection     nonCoulombicsSection;
    nonCoulombicsSection.processHeader(lineElements, *_engine);
    EXPECT_EQ(nonCoulombicsSection.getNonCoulombicType(), forceField::NonCoulombicType::LJ);

    lineElements = {"noncoulombics", "lj"};
    nonCoulombicsSection.processHeader(lineElements, *_engine);
    EXPECT_EQ(nonCoulombicsSection.getNonCoulombicType(), forceField::NonCoulombicType::LJ);

    lineElements = {"noncoulombics", "buckingham"};
    nonCoulombicsSection.processHeader(lineElements, *_engine);
    EXPECT_EQ(nonCoulombicsSection.getNonCoulombicType(), forceField::NonCoulombicType::BUCKINGHAM);

    lineElements = {"noncoulombics", "morse"};
    nonCoulombicsSection.processHeader(lineElements, *_engine);
    EXPECT_EQ(nonCoulombicsSection.getNonCoulombicType(), forceField::NonCoulombicType::MORSE);

    lineElements = {"noncoulombics", "lj", "dummy"};
    EXPECT_THROW(nonCoulombicsSection.processHeader(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"noncoulombics", "noValidType"};
    EXPECT_THROW(nonCoulombicsSection.processHeader(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, processSectionNonCoulombics)
{
    std::vector<std::string> lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    NonCoulombicsSection     nonCoulombicsSection;
    nonCoulombicsSection.setNonCoulombicType(forceField::NonCoulombicType::LJ);
    nonCoulombicsSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getNonCoulombicPairsVector().size(), 1);
    EXPECT_NO_THROW(
        dynamic_cast<const forceField::LennardJonesPair *>(_engine->getForceField().getNonCoulombicPairsVector()[0].get()));

    lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    nonCoulombicsSection.setNonCoulombicType(forceField::NonCoulombicType::BUCKINGHAM);
    EXPECT_NO_THROW(nonCoulombicsSection.processSection(lineElements, *_engine));
    EXPECT_EQ(_engine->getForceField().getNonCoulombicPairsVector().size(), 2);
    EXPECT_NO_THROW(
        dynamic_cast<const forceField::BuckinghamPair *>(_engine->getForceField().getNonCoulombicPairsVector()[0].get()));

    lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    nonCoulombicsSection.setNonCoulombicType(forceField::NonCoulombicType::MORSE);
    EXPECT_NO_THROW(nonCoulombicsSection.processSection(lineElements, *_engine));
    EXPECT_EQ(_engine->getForceField().getNonCoulombicPairsVector().size(), 3);
    EXPECT_NO_THROW(dynamic_cast<const forceField::MorsePair *>(_engine->getForceField().getNonCoulombicPairsVector()[0].get()));

    lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    nonCoulombicsSection.setNonCoulombicType(forceField::NonCoulombicType::LJ_9_12);
    EXPECT_THROW(nonCoulombicsSection.processSection(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, endedNormallyNonCoulombic)
{
    auto nonCoulombicsSection = NonCoulombicsSection();
    ASSERT_NO_THROW(nonCoulombicsSection.endedNormally(true));

    ASSERT_THROW_MSG(nonCoulombicsSection.endedNormally(false),
                     customException::ParameterFileException,
                     "Parameter file noncoulombics section ended abnormally!");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}