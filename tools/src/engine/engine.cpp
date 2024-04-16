/*****************************************************************************
<GPL_HEADER>

    PQ
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

#include "engine.hpp"

#include "analysisRunner.hpp"          // for AnalysisRunner
#include "toml.hpp"                    // for parse_error, operator<<, parse_file
#include "trajToComInFileReader.hpp"   // for TrajToComInFileReader

#include <cstdlib>     // for exit
#include <iostream>    // for operator<<, basic_ostream, ostream
#include <memory>      // for allocator, __shared_ptr_access
#include <stdexcept>   // for out_of_range

using namespace std;

void Engine::addAnalysisRunnerKeys()
{
    _analysisRunnerKeys.try_emplace("trajectoryToCenterOfMass",
                                    [this](const string_view &inputFilename)
                                    { _inputFileReaders.push_back(new TrajToComInFileReader(inputFilename)); });
}

Engine::Engine(const string_view &executableName, const string_view &inputFileName)
    : _executableName(executableName), _inputFilename(inputFileName)
{
}

void Engine::run()
{
    parseAnalysisRunners();

    for (const auto inputFileReader : _inputFileReaders)
    {
        _analysisRunners.push_back(&inputFileReader->read());
    }

    for (const auto analysisRunner : _analysisRunners)
    {
        analysisRunner->setup();
        analysisRunner->run();
    }
}

void Engine::parseAnalysisRunners()
{

    addAnalysisRunnerKeys();

    if (_executableName != "analysis")
        _analysisRunnerKeys.at(_executableName)(_inputFilename);
    else
    {
        toml::table tbl;
        try
        {
            tbl = toml::parse_file(_inputFilename);
        }
        catch (const toml::parse_error &err)
        {
            cerr << "Error parsing file '" << *err.source().path << "':\n"
                 << err.description() << "\n  (" << err.source().begin << ")\n";
            exit(-1);
        }

        try
        {
            const auto *runners = tbl["analysis"]["runners"].as_array();
            runners->for_each(
                [&](const toml::value<string> &runner)
                {
                    const auto *runnerName = runner.value_or("none");

                    _analysisRunnerKeys.at(runnerName)(_inputFilename);
                }

            );
        }
        catch (const std::out_of_range &err)
        {
            cerr << "Error parsing file '" << _inputFilename << "':\n"
                 << "  " << err.what() << "\n";
            ::exit(-1);
        }
    }
}