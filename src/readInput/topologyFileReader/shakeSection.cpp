#include "shakeSection.hpp"

#include "bondConstraint.hpp"   // for BondConstraint
#include "constraints.hpp"      // for Constraints
#include "engine.hpp"           // for Engine
#include "exceptions.hpp"       // for TopologyException
#include "simulationBox.hpp"    // for SimulationBox

#include <format>   // for format

using namespace readInput::topology;

/**
 * @brief processes the shake section of the topology file
 *
 * @details one line of the shake section contains 4 elements:
 * 1. atom index 1
 * 2. atom index 2
 * 3. target bond length
 * 4. linker (not used yet - not sure what it is for)
 *
 * @param line
 * @param engine
 *
 * @throws customException::TopologyException if number of elements in line is not 4
 * @throws customException::TopologyException if atom indices are the same (=same atoms)
 */
void ShakeSection::processSection(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 4)
        throw customException::TopologyException(
            std::format("Wrong number of arguments in topology file shake section at line {} - number of elements has to be 4!",
                        _lineNumber));

    auto atom1      = stoul(lineElements[0]);
    auto atom2      = stoul(lineElements[1]);
    auto bondLength = stod(lineElements[2]);
    // TODO: auto linker = lineElements[3];

    if (atom1 == atom2)
        throw customException::TopologyException(
            std::format("Topology file shake section at line {} - atoms cannot be the same!", _lineNumber));

    const auto [molecule1, atomIndex1] = engine.getSimulationBox().findMoleculeByAtomIndex(atom1);
    const auto [molecule2, atomIndex2] = engine.getSimulationBox().findMoleculeByAtomIndex(atom2);

    auto bondConstraint = constraints::BondConstraint(molecule1, molecule2, atomIndex1, atomIndex2, bondLength);

    engine.getConstraints().addBondConstraint(bondConstraint);
}

/**
 * @brief checks if shake sections ends normally
 *
 * @param endedNormal
 *
 * @throws customException::TopologyException if endedNormal is false
 */
void ShakeSection::endedNormally(bool endedNormal) const
{
    if (!endedNormal)
        throw customException::TopologyException(
            std::format("Topology file shake section at line {} - no end of section found!", _lineNumber));
}