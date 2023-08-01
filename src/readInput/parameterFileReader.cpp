#include "parameterFileReader.hpp"

using namespace std;
using namespace readInput;

/**
 * @brief constructor
 *
 * @param filename
 * @param engine
 */
ParameterFileReader::ParameterFileReader(const string &filename, engine::Engine &engine)
    : _filename(filename), _fp(filename), _engine(engine)
{
}

/**
 * @brief checks if reading topology file is needed
 *
 * @return true if shake is activated
 * @return true if force field is activated
 * @return false
 */
bool ParameterFileReader::isNeeded() const
{
    if (_engine.getConstraints().isActivated()) return true;

    if (_engine.getForceField().isActivated()) return true;

    return false;
}