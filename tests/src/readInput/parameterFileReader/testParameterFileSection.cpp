#include "exceptions.hpp"
#include "testTopologySection.hpp"
#include "topologySection.hpp"

using namespace ::testing;
using namespace readInput::topology;

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}