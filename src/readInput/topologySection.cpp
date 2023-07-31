#include "topologySection.hpp"

#include "bondConstraint.hpp"
#include "exceptions.hpp"
#include "stringUtilities.hpp"

using namespace std;
using namespace readInput;

/**
 * @brief general process function for topology sections
 *
 * @param line
 * @param engine
 */
void TopologySection::process(vector<string> &lineElements, engine::Engine &engine)
{
    string line;
    auto   endedNormal = false;

    while (getline(*_fp, line))
    {
        line         = StringUtilities::removeComments(line, "#");
        lineElements = StringUtilities::splitString(line);

        if (lineElements.empty())
        {
            ++_lineNumber;
            continue;
        }

        if (StringUtilities::to_lower_copy(lineElements[0]) == "end")
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

/**
 * @brief processes the shake section of the topology file
 *
 * @param line
 * @param engine
 */
void ShakeSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 4)
        throw customException::TopologyException("Wrong number of arguments in topology file shake section at line " +
                                                 to_string(_lineNumber) + " - number of elements has to be 4!");

    auto atom1      = stoul(lineElements[0]);
    auto atom2      = stoul(lineElements[1]);
    auto bondLength = stod(lineElements[2]);
    // TODO: auto linker = lineElements[3];

    if (atom1 == atom2)
        throw customException::TopologyException("Topology file shake section at line " + to_string(_lineNumber) +
                                                 " - atoms cannot be the same!");

    auto &&[molecule1, atomIndex1] = engine.getSimulationBox().findMoleculeByAtomIndex(atom1);
    auto &&[molecule2, atomIndex2] = engine.getSimulationBox().findMoleculeByAtomIndex(atom2);

    auto bondConstraint = constraints::BondConstraint(molecule1, molecule2, atomIndex1, atomIndex2, bondLength);

    engine.getConstraints().addBondConstraint(bondConstraint);
}

/**
 * @brief checks if shake sections ends normally
 *
 * @param endedNormal
 */
void ShakeSection::endedNormally(bool endedNormal) const
{
    if (!endedNormal)
        throw customException::TopologyException("Topology file shake section at line " + to_string(_lineNumber) +
                                                 " - no end of section found!");
}

/**
 * @brief processes the bond section of the topology file
 *
 * @param line
 * @param engine
 */
void BondSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 3 && lineElements.size() != 4)
        throw customException::TopologyException("Wrong number of arguments in topology file bond section at line " +
                                                 to_string(_lineNumber) + " - number of elements has to be 3 or 4!");

    auto atom1    = stoul(lineElements[0]);
    auto atom2    = stoul(lineElements[1]);
    auto bondType = stoul(lineElements[2]);
    // TODO: auto linker = lineElements[3];

    if (atom1 == atom2)
        throw customException::TopologyException("Topology file shake section at line " + to_string(_lineNumber) +
                                                 " - atoms cannot be the same!");

    auto &&[molecule1, atomIndex1] = engine.getSimulationBox().findMoleculeByAtomIndex(atom1);
    auto &&[molecule2, atomIndex2] = engine.getSimulationBox().findMoleculeByAtomIndex(atom2);

    auto bondForceField = forceField::BondForceField(molecule1, molecule2, atomIndex1, atomIndex2, bondType);

    engine.getForceField().addBond(bondForceField);
}

/**
 * @brief checks if bond sections ends normally
 *
 * @param endedNormal
 */
void BondSection::endedNormally(bool endedNormal) const
{
    if (!endedNormal)
        throw customException::TopologyException("Topology file bond section at line " + to_string(_lineNumber) +
                                                 " - no end of section found!");
}

/**
 * @brief processes the angle section of the topology file
 *
 * @param line
 * @param engine
 */
void AngleSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 4 && lineElements.size() != 5)
        throw customException::TopologyException("Wrong number of arguments in topology file angle section at line " +
                                                 to_string(_lineNumber) + " - number of elements has to be 4 or 5!");

    auto atom1     = stoul(lineElements[0]);
    auto atom2     = stoul(lineElements[1]);
    auto atom3     = stoul(lineElements[2]);
    auto angleType = stoul(lineElements[3]);

    if (atom1 == atom2 || atom1 == atom3 || atom2 == atom3)
        throw customException::TopologyException("Topology file angle section at line " + to_string(_lineNumber) +
                                                 " - atoms cannot be the same!");

    auto &&[molecule1, atomIndex1] = engine.getSimulationBox().findMoleculeByAtomIndex(atom1);
    auto &&[molecule2, atomIndex2] = engine.getSimulationBox().findMoleculeByAtomIndex(atom2);
    auto &&[molecule3, atomIndex3] = engine.getSimulationBox().findMoleculeByAtomIndex(atom3);

    auto angleForceField =
        forceField::AngleForceField({molecule1, molecule2, molecule3}, {atomIndex1, atomIndex2, atomIndex3}, angleType);

    engine.getForceField().addAngle(angleForceField);
}

/**
 * @brief checks if angle sections ends normally
 *
 * @param endedNormal
 */
void AngleSection::endedNormally(bool endedNormal) const
{
    if (!endedNormal)
        throw customException::TopologyException("Topology file angle section at line " + to_string(_lineNumber) +
                                                 " - no end of section found!");
}

/**
 * @brief processes the dihedral section of the topology file
 *
 * @param line
 * @param engine
 */
void DihedralSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 5 && lineElements.size() != 6)
        throw customException::TopologyException("Wrong number of arguments in topology file dihedral section at line " +
                                                 to_string(_lineNumber) + " - number of elements has to be 5 or 6!");

    auto atom1        = stoul(lineElements[0]);
    auto atom2        = stoul(lineElements[1]);
    auto atom3        = stoul(lineElements[2]);
    auto atom4        = stoul(lineElements[3]);
    auto dihedralType = stoul(lineElements[4]);

    if (atom1 == atom2 || atom1 == atom3 || atom1 == atom4 || atom2 == atom3 || atom2 == atom4 || atom3 == atom4)
        throw customException::TopologyException("Topology file dihedral section at line " + to_string(_lineNumber) +
                                                 " - atoms cannot be the same!");

    auto &&[molecule1, atomIndex1] = engine.getSimulationBox().findMoleculeByAtomIndex(atom1);
    auto &&[molecule2, atomIndex2] = engine.getSimulationBox().findMoleculeByAtomIndex(atom2);
    auto &&[molecule3, atomIndex3] = engine.getSimulationBox().findMoleculeByAtomIndex(atom3);
    auto &&[molecule4, atomIndex4] = engine.getSimulationBox().findMoleculeByAtomIndex(atom4);

    auto dihedralForceField = forceField::DihedralForceField(
        {molecule1, molecule2, molecule3, molecule4}, {atomIndex1, atomIndex2, atomIndex3, atomIndex4}, dihedralType);

    engine.getForceField().addDihedral(dihedralForceField);
}

/**
 * @brief checks if dihedral sections ends normally
 *
 * @param endedNormal
 */
void DihedralSection::endedNormally(bool endedNormal) const
{
    if (!endedNormal)
        throw customException::TopologyException("Topology file dihedral section at line " + to_string(_lineNumber) +
                                                 " - no end of section found!");
}

/**
 * @brief processes the improper section of the topology file
 *
 * @param line
 * @param engine
 */
void ImproperDihedralSection::processSection(vector<string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 5 && lineElements.size() != 6)
        throw customException::TopologyException("Wrong number of arguments in topology file improper dihedral section at line " +
                                                 to_string(_lineNumber) + " - number of elements has to be 5 or 6!");

    auto atom1                = stoul(lineElements[0]);
    auto atom2                = stoul(lineElements[1]);
    auto atom3                = stoul(lineElements[2]);
    auto atom4                = stoul(lineElements[3]);
    auto improperDihedralType = stoul(lineElements[4]);

    if (atom1 == atom2 || atom1 == atom3 || atom1 == atom4 || atom2 == atom3 || atom2 == atom4 || atom3 == atom4)
        throw customException::TopologyException("Topology file improper dihedral section at line " + to_string(_lineNumber) +
                                                 " - atoms cannot be the same!");

    auto &&[molecule1, atomIndex1] = engine.getSimulationBox().findMoleculeByAtomIndex(atom1);
    auto &&[molecule2, atomIndex2] = engine.getSimulationBox().findMoleculeByAtomIndex(atom2);
    auto &&[molecule3, atomIndex3] = engine.getSimulationBox().findMoleculeByAtomIndex(atom3);
    auto &&[molecule4, atomIndex4] = engine.getSimulationBox().findMoleculeByAtomIndex(atom4);

    auto improperDihedralForceField = forceField::DihedralForceField(
        {molecule1, molecule2, molecule3, molecule4}, {atomIndex1, atomIndex2, atomIndex3, atomIndex4}, improperDihedralType);

    engine.getForceField().addImproperDihedral(improperDihedralForceField);
}

/**
 * @brief checks if improper dihedral sections ends normally
 *
 * @param endedNormal
 */
void ImproperDihedralSection::endedNormally(bool endedNormal) const
{
    if (!endedNormal)
        throw customException::TopologyException("Topology file improper dihedral section at line " + to_string(_lineNumber) +
                                                 " - no end of section found!");
}