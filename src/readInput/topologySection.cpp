#include "topologySection.hpp"

#include "bondConstraint.hpp"
#include "exceptions.hpp"
#include "stringUtilities.hpp"

using namespace std;
using namespace readInput;

/**
 * @brief processes the shake section of the topology file
 *
 * @param line
 * @param engine
 */
void ShakeSection::process(vector<string> &, engine::Engine &engine)
{
    string line;
    auto   endedNormal = false;

    while (getline(*_fp, line))
    {
        line                    = StringUtilities::removeComments(line, "#");
        const auto lineElements = StringUtilities::splitString(line);

        if (lineElements.empty())
        {
            _lineNumber++;
            continue;
        }

        if (StringUtilities::to_lower_copy(lineElements[0]) == "end")
        {
            _lineNumber++;
            endedNormal = true;
            break;
        }

        if (lineElements.size() != 4)
            throw customException::TopologyException("Wrong number of arguments in topology file shake section at line " +
                                                     to_string(_lineNumber) + " - number of elements has to be 4!");

        auto atom1      = stoul(lineElements[0]);
        auto atom2      = stoul(lineElements[1]);
        auto bondLength = stod(lineElements[2]);
        // auto linker = lineElements[3];

        if (atom1 == atom2)
            throw customException::TopologyException("Topology file shake section at line " + to_string(_lineNumber) +
                                                     " - atoms cannot be the same!");

        auto &&[molecule1, atomIndex1] = engine.getSimulationBox().findMoleculeByAtomIndex(atom1);
        auto &&[molecule2, atomIndex2] = engine.getSimulationBox().findMoleculeByAtomIndex(atom2);

        auto bondConstraint = constraints::BondConstraint(molecule1, molecule2, atomIndex1, atomIndex2, bondLength);

        engine.getConstraints().addBondConstraint(bondConstraint);

        _lineNumber++;
    }

    if (!endedNormal) throw customException::TopologyException("Topology file shake section does not end with 'end' keyword!");
}