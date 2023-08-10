#include "inputFileParserIntegrator.hpp"

#include "exceptions.hpp"

#include <memory>

using namespace std;
using namespace readInput;
using namespace integrator;
using namespace customException;

/**
 * @brief Construct a new Input File Parser Integrator:: Input File Parser Integrator object
 *
 * @param engine
 */
InputFileParserIntegrator::InputFileParserIntegrator(engine::Engine &engine) : InputFileParser(engine)
{
    addKeyword(string("integrator"), bind_front(&InputFileParserIntegrator::parseIntegrator, this), false);
}

/**
 * @brief Parse the integrator used in the simulation
 *
 * @param lineElements
 *
 * @throws InputFileException if integrator is not valid - currently only velocity verlet is supported
 */
void InputFileParserIntegrator::parseIntegrator(const vector<string> &lineElements, const size_t lineNumber)
{
    checkCommand(lineElements, lineNumber);
    if (lineElements[2] == "v-verlet")
        _engine.makeIntegrator(VelocityVerlet());
    else
        throw InputFileException(format("Invalid integrator \"{}\" at line {} in input file", lineElements[2], lineNumber));
}