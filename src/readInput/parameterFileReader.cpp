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
    _parameterFileSections.push_back(make_unique<TypesSection>());
    _parameterFileSections.push_back(make_unique<BondSection>());
    _parameterFileSections.push_back(make_unique<AngleSection>());
    _parameterFileSections.push_back(make_unique<DihedralSection>());
    _parameterFileSections.push_back(make_unique<ImproperDihedralSection>());
    _parameterFileSections.push_back(make_unique<NonCoulombicsSection>());
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