#include "noseHooverSection.hpp"

#include "exceptions.hpp"

#include <format>   // for format
#include <string>   // for string
#include <vector>   // for vector

// TODO: not implemented yet

using namespace readInput::restartFile;

void NoseHooverSection::process(std::vector<std::string> &, engine::Engine &)
{
    throw customException::RstFileException(
        std::format("Error in line {}: Nose-Hoover section not implemented yet", _lineNumber));
}