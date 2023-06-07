#include "trajectoryToCom.hpp"

using namespace std;

void TrajToCom::setup()
{
    _configReader = ConfigurationReader(_xyzFilenames);
}

void TrajToCom::run()
{
    while (_configReader.nextFrame())
    {
        _frames.push_back(_configReader.getFrame());
    }
}