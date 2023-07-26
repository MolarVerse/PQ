#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace setup;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseTopologyFilename)
{
    vector<string> lineElements = {"topologyfilename", "=", ""};
    EXPECT_THROW(_inputFileReader->parseTopologyFilename(lineElements), customException::InputFileException);

    lineElements = {"topologyfilename", "=", "topology.txt"};
    EXPECT_THROW(_inputFileReader->parseTopologyFilename(lineElements), customException::InputFileException);

    lineElements = {"topologyfilename", "=", "data/topologyReader/topology.top"};
    _inputFileReader->parseTopologyFilename(lineElements);
    EXPECT_EQ(_engine.getSettings().getTopologyFilename(), "data/topologyReader/topology.top");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}