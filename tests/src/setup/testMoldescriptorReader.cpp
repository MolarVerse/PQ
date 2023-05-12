#include "testMoldesctripotReader.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace ::testing;

TEST_F(TestMoldescriptorReader, testFileNotFound)
{
    _engine._settings.setMoldescriptorFilename("data/moldescriptorReader/missingFile.txt");
    ASSERT_THROW(MoldescriptorReader reader(_engine), InputFileException);
}

TEST_F(TestMoldescriptorReader, testConstructor)
{
    _engine._settings.setMoldescriptorFilename("data/moldescriptorReader/moldescriptor.dat");
    ASSERT_NO_THROW(MoldescriptorReader reader(_engine));
}

TEST_F(TestMoldescriptorReader, testArguentsInMoldescriptor)
{
    _engine._settings.setMoldescriptorFilename("data/moldescriptorReader/moldescriptorWithOneWordLine.dat");
    MoldescriptorReader reader(_engine);
    ASSERT_THROW(reader.read(), MolDescriptorException);
}

TEST_F(TestMoldescriptorReader, testArgumentsinMoleculeSection)
{
    _engine._settings.setMoldescriptorFilename("data/moldescriptorReader/moldescriptorWithErrorInAtomArguments.dat");
    MoldescriptorReader reader(_engine);
    ASSERT_THROW(reader.read(), MolDescriptorException);

    _engine._settings.setMoldescriptorFilename("data/moldescriptorReader/moldescriptorWithErrorInAtomArguments2.dat");
    MoldescriptorReader reader2(_engine);
    ASSERT_THROW(reader2.read(), MolDescriptorException);

    _engine._settings.setMoldescriptorFilename("data/moldescriptorReader/moldescriptorWithErrorInMolArguments.dat");
    MoldescriptorReader reader3(_engine);
    ASSERT_THROW(reader3.read(), MolDescriptorException);
}

TEST_F(TestMoldescriptorReader, testMoldescriptorReader)
{
    _engine._settings.setMoldescriptorFilename("examples/setup/moldescriptor.dat");
    MoldescriptorReader reader(_engine);
    ASSERT_NO_THROW(reader.read());
}

TEST_F(TestMoldescriptorReader, testSpecialTypes)
{
    _engine._settings.setMoldescriptorFilename("examples/setup/moldescriptor.dat");
    readMolDescriptor(_engine);
    ASSERT_EQ(_engine.getSimulationBox().getWaterType(), 1);
    ASSERT_EQ(_engine.getSimulationBox().getAmmoniaType(), 2);
}

// TODO: build more large test cases with real simulation input to test everything!!!

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}