#include "noseHooverSection.hpp"

#include "exceptions.hpp"           // for RstFileException
#include "thermostatSettings.hpp"   // for ThermostatSettings

#include <format>   // for format
#include <string>   // for string
#include <vector>   // for vector

namespace engine
{
    class Engine;   // forward declaration
}

using readInput::restartFile::NoseHooverSection;

/**
 * @brief checks the number of arguments in the line
 *
 * @param lineElements all elements of the line
 *
 * @throws customException::RstFileException if the number of arguments is not correct
 */
void NoseHooverSection::process(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    if (4 != lineElements.size())
        throw customException::RstFileException(
            std::format("Error not enough arguments in line {} for a chi entry of the nose hoover thermostat", _lineNumber));

    auto [iterChi, chiIsInserted]   = settings::ThermostatSettings::addChi(stoul(lineElements[1]), stod(lineElements[2]));
    auto [iterZeta, zetaIsInserted] = settings::ThermostatSettings::addZeta(stoul(lineElements[1]), stod(lineElements[3]));

    if (!chiIsInserted || !zetaIsInserted)
        throw customException::RstFileException(
            std::format("Error in line {} in restart file; chi or zeta entry already exists", _lineNumber));
}