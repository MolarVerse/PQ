#include "inputFileReader.hpp"
#include "tomlExtensions.hpp"

#include <iostream>

using namespace std;

void InputFileReader::parseTomlFile()
{
    try
    {
        _tomlTable = toml::parse_file(_inputFilename);
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
}

vector<string> InputFileReader::parseXYZFiles()
{
    const auto input = _tomlTable["files"]["xyz"].as_array();

    return tomlExtensions::tomlArrayToVector<string>(input);
}

vector<size_t> InputFileReader::parseAtomIndices()
{
    const auto input = _tomlTable["system"]["atomIndices"].as_array();

    const auto atomIndicesToml = tomlExtensions::tomlArrayToVector<int64_t>(input);

    auto to_size_t = [](const int64_t &i)
    { return static_cast<int>(i); };

    vector<size_t> atomIndices(atomIndicesToml.size());

    transform(atomIndicesToml.begin(), atomIndicesToml.end(), atomIndices.begin(), to_size_t);

    return atomIndices;
}

size_t InputFileReader::parseNumberOfAtomsPerMolecule()
{
    return static_cast<size_t>(_tomlTable["system"]["numberOfAtomsPerMolecule"].value_or(0));
}