#include "exceptions.hpp"
#include "testMoldesctripotReader.hpp"
#include "throwWithMessage.hpp"

using namespace std;
using namespace ::testing;
using namespace readInput;
using namespace customException;

/**
 * @brief tests constructor of MoldescriptorReader
 *
 */
TEST_F(TestMoldescriptorReader, constructor)
{
    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/missingFile.txt");
    ASSERT_THROW_MSG(
        MoldescriptorReader reader(_engine), InputFileException, "\"data/moldescriptorReader/missingFile.txt\" File not found");

    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptor.dat");
    ASSERT_NO_THROW(MoldescriptorReader reader(_engine));
}

/**
 * @brief tests if line entry has at least 2 elements
 *
 */
TEST_F(TestMoldescriptorReader, argumentsInMoldescriptor)
{
    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptorWithOneWordLine.dat");
    MoldescriptorReader reader(_engine);
    ASSERT_THROW_MSG(reader.read(), MolDescriptorException, "Error in moldescriptor file at line 1");
}

/**
 * @brief tests number of entries for molecule section in moldescriptor
 *
 */
TEST_F(TestMoldescriptorReader, argumentsInMoleculeSection)
{
    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptorWithErrorInAtomArguments.dat");
    MoldescriptorReader reader(_engine);
    ASSERT_THROW_MSG(reader.read(), MolDescriptorException, "Error in moldescriptor file at line 4");

    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptorWithErrorInAtomArguments2.dat");
    MoldescriptorReader reader2(_engine);
    ASSERT_THROW_MSG(reader2.read(), MolDescriptorException, "Error in moldescriptor file at line 5");

    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptorWithErrorInMolArguments.dat");
    MoldescriptorReader reader3(_engine);
    ASSERT_THROW_MSG(reader3.read(), MolDescriptorException, "Error in moldescriptor file at line 3");
}

/**
 * @brief test reading of moldescriptor.dat
 *
 */
TEST_F(TestMoldescriptorReader, moldescriptorReader)
{
    _engine.getSettings().setMoldescriptorFilename("examples/setup/moldescriptor.dat");
    MoldescriptorReader reader(_engine);
    ASSERT_NO_THROW(reader.read());
}

/**
 * @brief test reading of special types
 *
 */
TEST_F(TestMoldescriptorReader, specialTypes)
{
    _engine.getSettings().setMoldescriptorFilename("examples/setup/moldescriptor.dat");
    readMolDescriptor(_engine);
    ASSERT_EQ(_engine.getSimulationBox().getWaterType(), 1);
    ASSERT_EQ(_engine.getSimulationBox().getAmmoniaType(), 2);
}

/**
 * @brief tests if there are to many atoms per moltype
 *
 */
TEST_F(TestMoldescriptorReader, toManyAtomsPerMoltype)
{
    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptorTooManyAtomsPerMoltype.dat");
    MoldescriptorReader reader2(_engine);
    ASSERT_THROW_MSG(reader2.read(), MolDescriptorException, "Error in moldescriptor file at line 3");
}

/**
 * @brief tests if non coulombic force field is activated but no global can der Waals parameter given
 *
 */
TEST_F(TestMoldescriptorReader, globalVdwTypes)
{
    _engine.getForceFieldPtr()->activateNonCoulombic();

    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptor_withGlobalVdwTypes.dat");
    EXPECT_NO_THROW(readMolDescriptor(_engine));

    _engine.getSettings().setMoldescriptorFilename("data/moldescriptorReader/moldescriptor_withMissingGlobalVdwTypes.dat");
    EXPECT_THROW_MSG(readMolDescriptor(_engine),
                     MolDescriptorException,
                     "Error in moldescriptor file at line 6 - force field noncoulombics is activated but no global can der Waals "
                     "parameter given");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}