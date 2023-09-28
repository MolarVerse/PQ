/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "../../include/input/commandLineArgs.hpp"
#include "engine.hpp"

#include <iostream>
#include <string>   // for string
#include <vector>   // for vector

using namespace std;

string getExecutableName()
{
#ifdef trajectoryToCenterOfMass
    return "trajectoryToCenterOfMass";
#else
    return "analysis";
#endif
}

int main(int argc, char **argv)
{
    // like in main.cpp of pimd_qmcf not best way TODO:
    vector<string> arguments(argv, argv + argc);
    auto           commandLineArgs = CommandLineArgs(argc, arguments);
    commandLineArgs.detectFlags();

    auto executableName = getExecutableName();

    auto engine = Engine(executableName, argv[1]);

    engine.run();

    return 0;
}