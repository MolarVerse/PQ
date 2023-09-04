#include "dihedralSection.hpp"

#include "constants.hpp"         // for _DEG_TO_RAD_
#include "dihedralType.hpp"      // for DihedralType
#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for ParameterFileException
#include "forceFieldClass.hpp"   // for ForceField

#include <format>   // for format

using namespace readInput::parameterFile;

/**
 * @brief processes one line of the dihedral section of the parameter file and adds the dihedral type to the force field
 *
 * @details The line is expected to have the following format:
 * 1. dihedralTypeId
 * 2. forceConstant
 * 3. periodicity
 * 4. phaseShift
 *
 * @param line
 * @param engine
 *
 * @throw customException::ParameterFileException if number of elements in line is not 3
 * @throw customException::ParameterFileException if periodicity is negative
 */
void DihedralSection::processSection(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 4)
        throw customException::ParameterFileException(
            std::format("Wrong number of arguments in parameter file angle section at line {} - number of elements has to be 4!",
                        _lineNumber));

    auto id            = stoul(lineElements[0]);
    auto forceConstant = stod(lineElements[1]);
    auto periodicity   = stod(lineElements[2]);
    auto phaseShift    = stod(lineElements[3]) * constants::_DEG_TO_RAD_;

    if (periodicity < 0.0)
        throw customException::ParameterFileException(
            std::format("Parameter file dihedral section at line {} - periodicity has to be positive!", _lineNumber));

    auto dihedralType = forceField::DihedralType(id, forceConstant, periodicity, phaseShift);

    engine.getForceField().addDihedralType(dihedralType);
}