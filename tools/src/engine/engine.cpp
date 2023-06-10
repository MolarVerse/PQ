#include "engine.hpp"
#include "trajectoryToCom.hpp"
#include "trajToComInFileReader.hpp"

#include <iostream>

using namespace std;

void Engine::addAnalysisRunnerKeys()
{
    _analysisRunnerKeys.try_emplace(
        "trajectoryToCenterOfMass", [this](const string_view &inputFilename)
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
}

void Engine::parseAnalysisRunners()
{

    try
    {
        auto pos = _executableName.find_last_of("/");
        _executableName = _executableName.substr(pos + 1);
    }
    catch (const std::exception &e)
    {
    }

    addAnalysisRunnerKeys();

    try
    {
        _analysisRunnerKeys.at(_executableName)(_inputFilename);
    }
    catch (const std::out_of_range &e)
    {
        toml::table tbl;
        try
        {
            tbl = toml::parse_file(_inputFilename);
        }
        catch (const toml::parse_error &err)
        {
            cerr
                << "Error parsing file '" << *err.source().path
                << "':\n"
                << err.description()
                << "\n  (" << err.source().begin << ")\n";
            exit(-1);
        }

        try
        {
            const toml::array *runners = tbl["analysis"]["runners"].as_array();
            runners->for_each(
                [this](auto &&runner)
                {
                string tmpRunner = runner.as_string()->value_or("");
                _analysisRunnerKeys.at(tmpRunner)(_inputFilename); });
        }
        catch (const std::out_of_range &e)
        {
        }
    }
}