#include "ringPolymerRestartFileReader.hpp"

#include "exceptions.hpp"            // for RingPolymerRestartFileException
#include "fileSettings.hpp"          // for FileSettings
#include "ringPolymerSettings.hpp"   // for RingPolymerSettings
#include "stringUtilities.hpp"       // for removeComments, splitString

using readInput::ringPolymer::RingPolymerRestartFileReader;

/**
 * @brief Reads a .rpmd.rst file sets the ring polymer beads in the engine
 *
 */
void RingPolymerRestartFileReader::read()
{
    std::string              line;
    std::vector<std::string> lineElements;
    int                      lineNumber = 0;

    const auto numberOfBeads = settings::RingPolymerSettings::getNumberOfBeads();

    for (size_t i = 0; i < numberOfBeads; i++)
    {
        for (auto &atom : _engine.getRingPolymerBeads()[i].getAtoms())
        {
            do
            {
                if (!getline(_fp, line))
                    throw customException::RingPolymerRestartFileException("Error reading ring polymer restart file");

                line         = utilities::removeComments(line, "#");
                lineElements = utilities::splitString(line);
                ++lineNumber;
            } while (lineElements.empty());

            if ((lineElements.size() != 21) && (lineElements.size() != 12))
                throw customException::RstFileException(
                    std::format("Error in line {}: Atom section must have 12 or 21 elements", lineNumber));

            atom->setPosition({stod(lineElements[3]), stod(lineElements[4]), stod(lineElements[5])});
            atom->setVelocity({stod(lineElements[6]), stod(lineElements[7]), stod(lineElements[8])});
            atom->setForce({stod(lineElements[9]), stod(lineElements[10]), stod(lineElements[11])});
        }
    }
}

/**
 * @brief wrapper function to construct a RingPolymerRestartFileReader object and call the read function
 *
 * @param engine
 */
void readInput::ringPolymer::readRingPolymerRestartFile(engine::RingPolymerEngine &engine)
{
    RingPolymerRestartFileReader reader(settings::FileSettings::getRingPolymerStartFileName(), engine);
    reader.read();
}