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

#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)

#include <memory>   // for allocator, shared_ptr
#include <string>   // for string, basic_string
#include <vector>   // for vector

#include "buckinghamPair.hpp"             // for BuckinghamPair
#include "engine.hpp"                     // for Engine
#include "exceptions.hpp"                 // for ParameterFileException
#include "forceFieldNonCoulomb.hpp"       // for ForceFieldNonCoulomb
#include "gtest/gtest.h"                  // for Message, TestPartResult, tes...
#include "lennardJonesPair.hpp"           // for LennardJonesPair
#include "morsePair.hpp"                  // for MorsePair
#include "nonCoulombPair.hpp"             // for NonCoulombPair
#include "nonCoulombPotential.hpp"        // for NonCoulombType, NonCoulombPo...
#include "nonCoulombicsSection.hpp"       // for NonCoulombicsSection
#include "parameterFileSection.hpp"       // for NonCoulombicsSection, parame...
#include "potential.hpp"                  // for Potential
#include "potentialSettings.hpp"          // for PotentialSettings
#include "testParameterFileSection.hpp"   // for TestParameterFileSection
#include "throwWithMessage.hpp"           // for ASSERT_THROW_MSG
#include "typeAliases.hpp"                // for pq::strings

using namespace input::parameterFile;
using namespace potential;
using namespace customException;
using namespace settings;

TEST_F(TestParameterFileSection, processSectionLennardJones)
{
    auto &potential = dynamic_cast<ForceFieldNonCoulomb &>(
        _engine->getPotential().getNonCoulombPotential()
    );

    pq::strings          lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    NonCoulombicsSection nonCoulombicsSection;
    nonCoulombicsSection.processLJ(lineElements, *_engine);
    EXPECT_EQ(potential.getNonCoulombPairsVector().size(), 1);

    const auto *pairVector = potential.getNonCoulombPairsVector()[0].get();
    const auto *pair       = dynamic_cast<const LennardJonesPair *>(pairVector);
    EXPECT_EQ(pair->getVanDerWaalsType1(), 0);
    EXPECT_EQ(pair->getVanDerWaalsType2(), 1);
    EXPECT_EQ(pair->getC6(), 1.22);
    EXPECT_EQ(pair->getC12(), 234.3);
    EXPECT_EQ(pair->getRadialCutOff(), 324.3);

    lineElements = {"0", "1", "1.22", "234.3"};
    nonCoulombicsSection.processLJ(lineElements, *_engine);
    EXPECT_EQ(potential.getNonCoulombPairsVector().size(), 2);

    const auto *pairVector2 = potential.getNonCoulombPairsVector()[1].get();
    auto       *pair2 = dynamic_cast<const LennardJonesPair *>(pairVector2);
    EXPECT_EQ(pair2->getVanDerWaalsType1(), 0);
    EXPECT_EQ(pair2->getVanDerWaalsType2(), 1);
    EXPECT_EQ(pair2->getC6(), 1.22);
    EXPECT_EQ(pair2->getC12(), 234.3);
    EXPECT_EQ(pair2->getRadialCutOff(), 12.5);

    lineElements = {"1", "2", "1.0", "0", "2", "3.3"};
    EXPECT_THROW(
        nonCoulombicsSection.processLJ(lineElements, *_engine),
        ParameterFileException
    );
}

TEST_F(TestParameterFileSection, processSectionBuckingham)
{
    auto &potential = dynamic_cast<ForceFieldNonCoulomb &>(
        _engine->getPotential().getNonCoulombPotential()
    );

    pq::strings lineElements = {"0", "1", "1.22", "234.3", "324.3", "435"};
    NonCoulombicsSection nonCoulombicsSection;
    nonCoulombicsSection.processBuckingham(lineElements, *_engine);
    EXPECT_EQ(potential.getNonCoulombPairsVector().size(), 1);

    const auto *pairVector = potential.getNonCoulombPairsVector()[0].get();
    const auto *pair       = dynamic_cast<const BuckinghamPair *>(pairVector);
    EXPECT_EQ(pair->getVanDerWaalsType1(), 0);
    EXPECT_EQ(pair->getVanDerWaalsType2(), 1);
    EXPECT_EQ(pair->getA(), 1.22);
    EXPECT_EQ(pair->getDRho(), 234.3);
    EXPECT_EQ(pair->getC6(), 324.3);
    EXPECT_EQ(pair->getRadialCutOff(), 435.0);

    lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    nonCoulombicsSection.processBuckingham(lineElements, *_engine);
    EXPECT_EQ(potential.getNonCoulombPairsVector().size(), 2);

    const auto *pairVector2 = potential.getNonCoulombPairsVector()[1].get();
    const auto *pair2       = dynamic_cast<const BuckinghamPair *>(pairVector2);
    EXPECT_EQ(pair2->getVanDerWaalsType1(), 0);
    EXPECT_EQ(pair2->getVanDerWaalsType2(), 1);
    EXPECT_EQ(pair2->getA(), 1.22);
    EXPECT_EQ(pair2->getDRho(), 234.3);
    EXPECT_EQ(pair2->getC6(), 324.3);
    EXPECT_EQ(pair2->getRadialCutOff(), 12.5);

    lineElements = {"1", "2", "1.0", "0", "2", "3.3", "345"};
    EXPECT_THROW(
        nonCoulombicsSection.processBuckingham(lineElements, *_engine),
        ParameterFileException
    );
}

TEST_F(TestParameterFileSection, processSectionMorse)
{
    auto &potential = dynamic_cast<ForceFieldNonCoulomb &>(
        _engine->getPotential().getNonCoulombPotential()
    );

    pq::strings lineElements = {"0", "1", "1.22", "234.3", "324.3", "435"};
    NonCoulombicsSection nonCoulombicsSection;
    nonCoulombicsSection.processMorse(lineElements, *_engine);
    EXPECT_EQ(potential.getNonCoulombPairsVector().size(), 1);
    auto *pair = dynamic_cast<const MorsePair *>(
        potential.getNonCoulombPairsVector()[0].get()
    );
    EXPECT_EQ(pair->getVanDerWaalsType1(), 0);
    EXPECT_EQ(pair->getVanDerWaalsType2(), 1);
    EXPECT_EQ(pair->getDissociationEnergy(), 1.22);
    EXPECT_EQ(pair->getWellWidth(), 234.3);
    EXPECT_EQ(pair->getEquilibriumDistance(), 324.3);
    EXPECT_EQ(pair->getRadialCutOff(), 435.0);

    lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    nonCoulombicsSection.processMorse(lineElements, *_engine);
    EXPECT_EQ(potential.getNonCoulombPairsVector().size(), 2);
    auto *pair2 = dynamic_cast<const MorsePair *>(
        potential.getNonCoulombPairsVector()[1].get()
    );
    EXPECT_EQ(pair2->getVanDerWaalsType1(), 0);
    EXPECT_EQ(pair2->getVanDerWaalsType2(), 1);
    EXPECT_EQ(pair2->getDissociationEnergy(), 1.22);
    EXPECT_EQ(pair2->getWellWidth(), 234.3);
    EXPECT_EQ(pair2->getEquilibriumDistance(), 324.3);
    EXPECT_EQ(pair2->getRadialCutOff(), 12.5);

    lineElements = {"1", "2", "1.0", "0", "2", "3.3", "345"};
    EXPECT_THROW(
        nonCoulombicsSection.processMorse(lineElements, *_engine),
        ParameterFileException
    );
}

TEST_F(TestParameterFileSection, processHeader)
{
    pq::strings          lineElements = {"noncoulombics"};
    NonCoulombicsSection nonCoulombicsSection;
    nonCoulombicsSection.processHeader(lineElements, *_engine);
    EXPECT_EQ(PotentialSettings::getNonCoulombType(), NonCoulombType::LJ);

    lineElements = {"noncoulombics", "lj"};
    nonCoulombicsSection.processHeader(lineElements, *_engine);
    EXPECT_EQ(PotentialSettings::getNonCoulombType(), NonCoulombType::LJ);

    lineElements = {"noncoulombics", "buckingham"};
    nonCoulombicsSection.processHeader(lineElements, *_engine);
    EXPECT_EQ(
        PotentialSettings::getNonCoulombType(),
        NonCoulombType::BUCKINGHAM
    );

    lineElements = {"noncoulombics", "morse"};
    nonCoulombicsSection.processHeader(lineElements, *_engine);
    EXPECT_EQ(PotentialSettings::getNonCoulombType(), NonCoulombType::MORSE);

    lineElements = {"noncoulombics", "lj", "dummy"};
    EXPECT_NO_THROW(nonCoulombicsSection.processHeader(lineElements, *_engine));

    lineElements = {"noncoulombics", "noValidType"};
    EXPECT_THROW(
        nonCoulombicsSection.processHeader(lineElements, *_engine),
        ParameterFileException
    );
}

TEST_F(TestParameterFileSection, processSectionNonCoulombics)
{
    auto &potential = dynamic_cast<ForceFieldNonCoulomb &>(
        _engine->getPotential().getNonCoulombPotential()
    );

    pq::strings          lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    NonCoulombicsSection nonCoulombicsSection;
    PotentialSettings::setNonCoulombType(NonCoulombType::LJ);
    nonCoulombicsSection.processSection(lineElements, *_engine);
    EXPECT_EQ(potential.getNonCoulombPairsVector().size(), 1);
    EXPECT_NO_THROW(
        dynamic_cast<const LennardJonesPair *>(
            potential.getNonCoulombPairsVector()[0].get()
        )
    );

    lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    PotentialSettings::setNonCoulombType(NonCoulombType::BUCKINGHAM);
    EXPECT_NO_THROW(nonCoulombicsSection.processSection(lineElements, *_engine)
    );
    EXPECT_EQ(potential.getNonCoulombPairsVector().size(), 2);
    EXPECT_NO_THROW(
        dynamic_cast<const BuckinghamPair *>(
            potential.getNonCoulombPairsVector()[0].get()
        )
    );

    lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    PotentialSettings::setNonCoulombType(NonCoulombType::MORSE);
    EXPECT_NO_THROW(nonCoulombicsSection.processSection(lineElements, *_engine)
    );
    EXPECT_EQ(potential.getNonCoulombPairsVector().size(), 3);
    EXPECT_NO_THROW(
        dynamic_cast<const MorsePair *>(
            potential.getNonCoulombPairsVector()[0].get()
        )
    );

    lineElements = {"0", "1", "1.22", "234.3", "324.3"};
    PotentialSettings::setNonCoulombType(NonCoulombType::LJ_9_12);
    EXPECT_THROW(
        nonCoulombicsSection.processSection(lineElements, *_engine),
        ParameterFileException
    );
}

TEST_F(TestParameterFileSection, endedNormallyNonCoulombic)
{
    auto nonCoulombicsSection = NonCoulombicsSection();
    ASSERT_NO_THROW(nonCoulombicsSection.endedNormally(true));

    ASSERT_THROW_MSG(
        nonCoulombicsSection.endedNormally(false),
        ParameterFileException,
        "Parameter file noncoulombics section ended abnormally!"
    );
}