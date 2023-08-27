#include "bondSection.hpp"

#include "engine.hpp"
#include "exceptions.hpp"
#include "forceField.hpp"

#include <format>   // for format

using namespace readInput::parameterFile;

/**
 * @brief processes one line of the bond section of the parameter file and adds the bond type to the force field
 *
 * @details The line is expected to have the following format:
 * 1. bondTypeId
 * 2. equilibriumDistance
 * 3. forceConstant
 *
 * @param line
 * @param engine
 *
 * @throw customException::ParameterFileException if number of elements in line is not 3
 * @throw customException::ParameterFileException if equilibrium distance is negative
 */
void BondSection::processSection(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 3)
        throw customException::ParameterFileException(
            std::format("Wrong number of arguments in parameter file bond section at line {} - number of elements has to be 3!",
                        _lineNumber));

    auto id                  = stoul(lineElements[0]);
    auto equilibriumDistance = stod(lineElements[1]);
    auto forceConstant       = stod(lineElements[2]);

    if (equilibriumDistance < 0.0)
        throw customException::ParameterFileException(
            std::format("Parameter file bond section at line {} - equilibrium distance has to be positive!", _lineNumber));

    auto bondType = forceField::BondType(id, equilibriumDistance, forceConstant);

    engine.getForceField().addBondType(bondType);
}