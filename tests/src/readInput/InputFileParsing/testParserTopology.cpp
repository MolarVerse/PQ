#include "exceptions.hpp"
#include "testInputFileReader.hpp"

using namespace std;
using namespace readInput;
using namespace ::testing;

TEST_F(TestInputFileReader, testParseTopologyFilename)
{
    vector<string> lineElements = {"topology_file", "=", ""};
    EXPECT_THROW(_inputFileReader->parseTopologyFilename(lineElements), customException::InputFileException);

    lineElements = {"topology_file", "=", "topology.txt"};
    EXPECT_THROW(_inputFileReader->parseTopologyFilename(lineElements), customException::InputFileException);

    lineElements = {"topology_file", "=", "data/topologyReader/topology.top"};
    _inputFileReader->parseTopologyFilename(lineElements);
    EXPECT_EQ(_engine.getSettings().getTopologyFilename(), "data/topologyReader/topology.top");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}