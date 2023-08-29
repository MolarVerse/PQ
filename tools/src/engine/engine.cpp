#include "engine.hpp"

#include "trajToComInFileReader.hpp"

#include <iostream>

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