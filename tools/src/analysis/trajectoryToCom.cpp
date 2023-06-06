#include "trajectoryToCom.hpp"

using namespace std;

void TrajectoryToCom::setup()
{
    _configReader = ConfigurationReader(_xyzFilenames);
}

void TrajectoryToCom::run()
{
    while (_configReader.nextFrame())
    {
        _frames.push_back(_configReader.getFrame());
    }
}