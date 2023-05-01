#include <iostream>

#include "rstFileReader.hpp"
#include "simulationBox.hpp"

using namespace std;

void print_test(SimulationBox &simulationBox)
{
    cout << simulationBox._atomtype.size() << endl;
}

int main()
{
    auto settings = *new Settings;
    auto simulationBox = read_rst("h2o-qmcf.rst", settings);

    cout << "Step count: " << settings.getStepCount() << endl;

    print_test(*simulationBox);

    return 0;
}