#include <iostream>

#include "rstFileReader.hpp"
#include "simulationBox.hpp"

using namespace std;

int main()
{
    auto settings = *new Settings;
    auto simulationBox = read_rst("h2o-qmcf.rst", settings);

    cout << "Step count: " << settings.getStepCount() << endl;
    return 0;
}