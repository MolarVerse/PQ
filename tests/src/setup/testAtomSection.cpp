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

namespace Setup::RstFileReader
{
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

    TEST_F(TestAtomSection, testNumberOfArgumentsWithinMolecule)
    {
        auto line = vector<string>(21);
        line[2] = "1";
        for (int i = 3; i < 21; ++i)
            line[i] = "1.0";

        string filename = "data/atomSection/testNumberOfArgumentsWithinMolecule.rst";

        auto molecule = Molecule(1);
        molecule.setNumberOfAtoms(3);
        _engine._simulationBox._moleculeTypes.push_back(molecule);

        ifstream fp(filename);
        _section->_fp = &fp;

        ASSERT_THROW(_section->process(line, _engine), RstFileException);
    }

    TEST_F(TestAtomSection, testProcess)
    {
        auto line = vector<string>(21);
        line[2] = "1";
        for (int i = 3; i < 21; ++i)
            line[i] = "1.0";

        string filename = "data/atomSection/testProcess.rst";

        auto molecule = Molecule(1);
        molecule.setNumberOfAtoms(3);
        _engine._simulationBox._moleculeTypes.push_back(molecule);

        auto molecule2 = Molecule(2);
        molecule2.setNumberOfAtoms(4);
        _engine._simulationBox._moleculeTypes.push_back(molecule2);

        ifstream fp(filename);
        _section->_fp = &fp;

        _section->process(line, _engine);

        line = vector<string>(21);
        line[2] = "2";
        for (int i = 3; i < 21; ++i)
            line[i] = "1.0";

        _section->process(line, _engine);

        line = vector<string>(21);
        line[2] = "1";
        for (int i = 3; i < 21; ++i)
            line[i] = "1.0";

        _section->process(line, _engine);

        EXPECT_EQ(_engine._simulationBox._molecules.size(), 3);

        EXPECT_EQ(_engine._simulationBox._molecules[0].getMoltype(), 1);
        EXPECT_EQ(_engine._simulationBox._molecules[0].getNumberOfAtoms(), 3);

        EXPECT_EQ(_engine._simulationBox._molecules[1].getMoltype(), 2);
        EXPECT_EQ(_engine._simulationBox._molecules[1].getNumberOfAtoms(), 4);

        EXPECT_EQ(_engine._simulationBox._molecules[2].getMoltype(), 1);
        EXPECT_EQ(_engine._simulationBox._molecules[2].getNumberOfAtoms(), 3);
    }

    TEST_F(TestAtomSection, testProcessAtomLine)
    {
        Molecule molecule(1);

        auto line = vector<string>(21);
        line[0] = "Ar";
        for (int i = 3; i < 21; ++i)
            line[i] = to_string(i + i / 10.0);

        static_cast<AtomSection *>(_section)->processAtomLine(line, molecule);

        ASSERT_THAT(molecule.getAtomPosition(0), ElementsAre(stod(line[3]), stod(line[4]), stod(line[5])));
        ASSERT_THAT(molecule.getAtomVelocity(0), ElementsAre(stod(line[6]), stod(line[7]), stod(line[8])));
        ASSERT_THAT(molecule.getAtomForce(0), ElementsAre(stod(line[9]), stod(line[10]), stod(line[11])));
        ASSERT_THAT(molecule.getAtomPositionOld(0), ElementsAre(stod(line[12]), stod(line[13]), stod(line[14])));
        ASSERT_THAT(molecule.getAtomVelocityOld(0), ElementsAre(stod(line[15]), stod(line[16]), stod(line[17])));
        ASSERT_THAT(molecule.getAtomForceOld(0), ElementsAre(stod(line[18]), stod(line[19]), stod(line[20])));

        ASSERT_EQ(molecule.getAtomTypeName(0), line[0]);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}