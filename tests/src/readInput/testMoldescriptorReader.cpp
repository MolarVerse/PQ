#include "exceptions.hpp"
#include "testMoldesctripotReader.hpp"

using namespace std;
using namespace ::testing;
using namespace setup;
using namespace customException;

TEST_F(TestMoldescriptorReader, testFileNotFound)
{
    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/missingFile.txt");
    ASSERT_THROW(MoldescriptorReader reader(_engine), InputFileException);
}

TEST_F(TestMoldescriptorReader, testConstructor)
{
    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptor.dat");
    ASSERT_NO_THROW(MoldescriptorReader reader(_engine));
}

TEST_F(TestMoldescriptorReader, testArguentsInMoldescriptor)
{
    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptorWithOneWordLine.dat");
    MoldescriptorReader reader(_engine);
    ASSERT_THROW(reader.read(), MolDescriptorException);
}

TEST_F(TestMoldescriptorReader, testArgumentsinMoleculeSection)
{
    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptorWithErrorInAtomArguments.dat");
    MoldescriptorReader reader(_engine);
    ASSERT_THROW(reader.read(), MolDescriptorException);

    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptorWithErrorInAtomArguments2.dat");
    MoldescriptorReader reader2(_engine);
    ASSERT_THROW(reader2.read(), MolDescriptorException);

    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptorWithErrorInMolArguments.dat");
    MoldescriptorReader reader3(_engine);
    ASSERT_THROW(reader3.read(), MolDescriptorException);
}

/**
 * @brief test reading of moldescriptor.dat
 *
 * TODO: add more tests to check if the data is read correctly
 *
 */
TEST_F(TestMoldescriptorReader, testMoldescriptorReader)
{
    _engine.getSettings().setMoldescriptorFilename("examples/setup/moldescriptor.dat");
    MoldescriptorReader reader(_engine);
    ASSERT_NO_THROW(reader.read());
}

TEST_F(TestMoldescriptorReader, testSpecialTypes)
{
    _engine.getSettings().setMoldescriptorFilename("examples/setup/moldescriptor.dat");
    readMolDescriptor(_engine);
    ASSERT_EQ(_engine.getSimulationBox().getWaterType(), 1);
    ASSERT_EQ(_engine.getSimulationBox().getAmmoniaType(), 2);
}

TEST_F(TestMoldescriptorReader, testToManyAtomsPerMoltype)
{
    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptorTooManyAtomsPerMoltype.dat");
    MoldescriptorReader reader2(_engine);
    ASSERT_THROW(reader2.read(), MolDescriptorException);
}

// TODO: build more large test cases with real simulation input to test everything!!!

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}