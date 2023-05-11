#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <string>
#include <cassert>
#include <fstream>
#include <filesystem>

#include "testRstFileSection.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace testing;

TEST_F(TestAtomSection, testKeyword)
{
    EXPECT_EQ(_section->keyword(), "");
}

TEST_F(TestAtomSection, testIsHeader)
{
    EXPECT_FALSE(_section->isHeader());
}

TEST_F(TestAtomSection, testNumberOfArguments)
{
    for (int i = 0; i < 25; ++i)
        if (i != 21)
        {
            auto line = vector<string>(i);
            ASSERT_THROW(_section->process(line, _engine), RstFileException);
        }
}

TEST_F(TestAtomSection, testMoltypeNotFound)
{
    auto line = vector<string>(21);
    line[2] = "1";
    ASSERT_THROW(_section->process(line, _engine), RstFileException);
}

TEST_F(TestAtomSection, testNotEnoughAtomsInMolecule)
{
    auto line = vector<string>(21);
    line[2] = "1";
    for (int i = 3; i < 21; ++i)
        line[i] = "1.0";

    string filename = "data/atomSection/testNotEnoughAtomsInMolecule.rst";

    auto molecule = Molecule(1);
    molecule.setNumberOfAtoms(3);
    _engine._simulationBox._moleculeTypes.push_back(molecule);

    ifstream fp(filename);
    _section->_fp = &fp;

    ASSERT_THROW(_section->process(line, _engine), RstFileException);

    line[2] = "1";

    string filename2 = "data/atomSection/testNotEnoughAtomsInMolecule2.rst";
    ifstream fp2(filename2);
    _section->_fp = &fp2;

    ASSERT_THROW(_section->process(line, _engine), RstFileException);
}