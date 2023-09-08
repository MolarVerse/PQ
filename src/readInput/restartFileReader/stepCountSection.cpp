#include "stepCountSection.hpp"

#include "engine.hpp"       // for Engine
#include "exceptions.hpp"   // for RstFileException
#include "timings.hpp"      // for Timings

#include <cstddef>   // for size_t
#include <format>    // for format
#include <string>    // for string, stoi
#include <vector>    // for vector

using namespace readInput::restartFile;

/**
 * @brief processes the step count section of the rst file
 *
 * @details The step count section is a header section and must have 2 elements:
 * 1. keyword "step"
 * 2. step count
 *
 * @param lineElements all elements of the line
 * @param engine object containing the engine
 *
 * @throws RstFileException if the number of elements in the line is not 2
 * @throws RstFileException if the step count is negative
 */
void StepCountSection::process(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 2)
        throw customException::RstFileException(
            std::format("Error in line {}: Step count section must have 2 elements", _lineNumber));

    auto stepCount = stoi(lineElements[1]);

    if (stepCount < 0)
        throw customException::RstFileException(std::format("Error in line {}: Step count must be positive", _lineNumber));

    engine.getTimings().setStepCount(size_t(stepCount));
}