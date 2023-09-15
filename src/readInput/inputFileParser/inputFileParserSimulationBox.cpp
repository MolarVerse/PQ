#include "inputFileParserSimulationBox.hpp"

#include "engine.hpp"                  // for Engine
#include "exceptions.hpp"              // for InputFileException, customException
#include "potentialSettings.hpp"       // for PotentialSettings
#include "simulationBox.hpp"           // for SimulationBox
#include "simulationBoxSettings.hpp"   // for setDensitySet

#include <cstddef>      // for size_t
#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

using namespace readInput;

/**
 * @brief Construct a new Input File Parser Simulation Box:: Input File Parser Simulation Box object
 *
 * @details following keywords are added to the _keywordFuncMap, _keywordRequiredMap and _keywordCountMap:
 * 1) rcoulomb <double>
 * 2) density <double>
 *
 * @param engine
 */
InputFileParserSimulationBox::InputFileParserSimulationBox(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(std::string("rcoulomb"), bind_front(&InputFileParserSimulationBox::parseCoulombRadius, this), false);
    addKeyword(std::string("density"), bind_front(&InputFileParserSimulationBox::parseDensity, this), false);
    addKeyword(std::string("init_velocities"), bind_front(&InputFileParserSimulationBox::parseInitializeVelocities, this), false);
}

/**
 * @brief parses the coulomb cutoff radius
 *
 * @details default value is 12.5
 *
 * @param lineElements
 *
 * @throw customException::InputFileException if the cutoff radius is negative
 */
void InputFileParserSimulationBox::parseCoulombRadius(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto cutOff = stod(lineElements[2]);

    if (cutOff < 0.0)
        throw customException::InputFileException(
            format("Coulomb radius cutoff must be positive - \"{}\" at line {} in input file", lineElements[2], lineNumber));

    _engine.getSimulationBox().setCoulombRadiusCutOff(cutOff);
    settings::PotentialSettings::setCoulombRadiusCutOff(cutOff);
}

/**
 * @brief parse density of simulation and set it in simulation box
 *
 * @details set in simulationBoxSettings if density is set to put warning if both density and box size are set
 *
 * @param lineElements
 *
 * @throw customException::InputFileException if the density is negative
 */
void InputFileParserSimulationBox::parseDensity(const std::vector<std::string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);

    const auto density = stod(lineElements[2]);

    if (density < 0.0)
        throw customException::InputFileException(std::format("Density must be positive - density = {}", density));

    settings::SimulationBoxSettings::setDensitySet(true);
    _engine.getSimulationBox().setDensity(density);
}

/**
 * @brief parse if velocities should be initialized with maxwell boltzmann distribution
 *
 * @details possible options are:
 * 1) true
 * 2) false (default)
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserSimulationBox::parseInitializeVelocities(const std::vector<std::string> &lineElements,
                                                             const size_t                    lineNumber)
{
    checkCommand(lineElements, lineNumber);

    if (lineElements[2] == "true")
        settings::SimulationBoxSettings::setInitializeVelocities(true);
    else if (lineElements[2] == "false")
        settings::SimulationBoxSettings::setInitializeVelocities(false);
    else
        throw customException::InputFileException(std::format(
            "Invalid value for initialize velocities - \"{}\" at line {} in input file. Possible options are: true, false",
            lineElements[2],
            lineNumber));
}