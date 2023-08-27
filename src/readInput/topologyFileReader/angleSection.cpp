#include "angleSection.hpp"

#include "angleForceField.hpp"   // for BondForceField
#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for TopologyException

#include <format>   // for format
#include <string>   // for string, allocator
#include <vector>   // for vector

using namespace readInput::topology;

/**
 * @brief processes the angle section of the topology file
 *
 * @details one line consists of 4 or 5 elements:
 * 1. atom index 1
 * 2. atom index 2 (center atom)
 * 3. atom index 3
 * 4. angle type
 * 5. linker marked with a '*' (optional)
 *
 * @param line
 * @param engine
 *
 * @throws customException::TopologyException if number of elements in line is not 4 or 5
 * @throws customException::TopologyException if atom indices are the same (=same atoms)
 * @throws customException::TopologyException if fifth element is not a '*'
 */
void AngleSection::processSection(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 4 && lineElements.size() != 5)
        throw customException::TopologyException(std::format(
            "Wrong number of arguments in topology file angle section at line {} - number of elements has to be 4 or 5!",
            _lineNumber));

    auto atom1     = stoul(lineElements[0]);
    auto atom2     = stoul(lineElements[1]);
    auto atom3     = stoul(lineElements[2]);
    auto angleType = stoul(lineElements[3]);
    auto isLinker  = false;

    if (5 == lineElements.size())
    {
        if (lineElements[4] == "*")
            isLinker = true;
        else
            throw customException::TopologyException(
                std::format("Fifth entry in topology file in angle section has to be a \'*\' or empty at line {}!", _lineNumber));
    }

    if (atom1 == atom2 || atom1 == atom3 || atom2 == atom3)
        throw customException::TopologyException(
            std::format("Topology file angle section at line {} - atoms cannot be the same!", _lineNumber));

    const auto [molecule1, atomIndex1] = engine.getSimulationBox().findMoleculeByAtomIndex(atom1);
    const auto [molecule2, atomIndex2] = engine.getSimulationBox().findMoleculeByAtomIndex(atom2);
    const auto [molecule3, atomIndex3] = engine.getSimulationBox().findMoleculeByAtomIndex(atom3);

    auto angleForceField =
        forceField::AngleForceField({molecule2, molecule1, molecule3}, {atomIndex2, atomIndex1, atomIndex3}, angleType);
    angleForceField.setIsLinker(isLinker);

    engine.getForceField().addAngle(angleForceField);
}

/**
 * @brief checks if angle sections ends normally
 *
 * @param endedNormal
 *
 * @throws customException::TopologyException if endedNormal is false
 */
void AngleSection::endedNormally(bool endedNormal) const
{
    if (!endedNormal)
        throw customException::TopologyException(
            std::format("Topology file angle section at line {} - no end of section found!", _lineNumber));
}