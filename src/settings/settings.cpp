#include <stdexcept>
#include <string>

#include "settings.hpp"

using namespace std;

Settings::Settings() {}
Settings::~Settings() {}

int Settings::getStepCount() { return _stepcount; }
void Settings::setStepCount(int stepcount)
{
    if(stepcount < 0) throw range_error("Step count in restart file must be positive - stepcount = " + to_string(stepcount) );
    _stepcount = stepcount;
}