#include "angleSection.hpp"

#include "angleType.hpp"    // for AngleType
#include "constants.hpp"    // for _DEG_TO_RAD_
#include "engine.hpp"       // for Engine
#include "exceptions.hpp"   // for ParameterFileException
#include "forceField.hpp"   // for ForceField

#include <format>   // for format

using namespace readInput::parameterFile;

/**
 * @brief processes one line of the angle section of the parameter file and adds the angle type to the force field
 *
 * @details The line is expected to have the following format:
 * 1. angleTypeId
 * 2. equilibriumAngle
 * 3. forceConstant
 *
 * @param line
 * @param engine
 *
 * @throw customException::ParameterFileException if number of elements in line is not 3
 */
void AngleSection::processSection(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 3)
        throw customException::ParameterFileException(
            std::format("Wrong number of arguments in parameter file angle section at line {} - number of elements has to be 3!",
                        _lineNumber));

    auto id               = stoul(lineElements[0]);
    auto equilibriumAngle = stod(lineElements[1]) * constants::_DEG_TO_RAD_;
    auto forceConstant    = stod(lineElements[2]);

    auto angleType = forceField::AngleType(id, equilibriumAngle, forceConstant);

    engine.getForceField().addAngleType(angleType);
}