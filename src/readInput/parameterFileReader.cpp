#include "parameterFileReader.hpp"

using namespace std;
using namespace readInput::parameterFile;

/**
 * @brief constructor
 *
 * @param filename
 * @param engine
 */
ParameterFileReader::ParameterFileReader(const string &filename, engine::Engine &engine)
    : _filename(filename), _fp(filename), _engine(engine)
{
    _parameterFileSections.push_back(new TypesSection());
    _parameterFileSections.push_back(new BondSection());
    _parameterFileSections.push_back(new AngleSection());
    _parameterFileSections.push_back(new DihedralSection());
    _parameterFileSections.push_back(new ImproperDihedralSection());
}

/**
 * @brief Destructor
 */
ParameterFileReader::~ParameterFileReader()
{
    for (auto *section : _parameterFileSections)
        delete section;
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