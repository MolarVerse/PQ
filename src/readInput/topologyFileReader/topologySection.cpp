#include "topologySection.hpp"

#include "angleForceField.hpp"      // for AngleForceField
#include "bondConstraint.hpp"       // for BondConstraint
#include "bondForceField.hpp"       // for BondForceField
#include "constraints.hpp"          // for Constraints
#include "dihedralForceField.hpp"   // for DihedralForceField
#include "engine.hpp"               // for Engine
#include "exceptions.hpp"           // for TopologyException
#include "forceField.hpp"           // for ForceField
#include "simulationBox.hpp"        // for SimulationBox
#include "stringUtilities.hpp"      // for removeComments, splitString, toLow...

#include <cstddef>   // for size_t
#include <format>    // for format
#include <istream>   // for basic_istream, ifstream

using namespace readInput::topology;

/**
 * @brief general process function for topology sections
 *
 * @details Reads the topology file line by line and calls the processSection function for each line until the "end" keyword is
 * found. At the end of the section the endedNormally function is called, which checks if the "end" keyword was found.
 *
 * @param line
 * @param engine
 */
void TopologySection::process(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    std::string line;
    auto        endedNormal = false;

    while (getline(*_fp, line))
    {
        line         = utilities::removeComments(line, "#");
        lineElements = utilities::splitString(line);

        if (lineElements.empty())
        {
            ++_lineNumber;
            continue;
        }

        if (utilities::toLowerCopy(lineElements[0]) == "end")
        {
            ++_lineNumber;
            endedNormal = true;
            break;
        }

        processSection(lineElements, engine);

        ++_lineNumber;
    }

    endedNormally(endedNormal);
}