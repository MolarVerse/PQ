#ifndef _INPUTFILEREADER_HPP_

#define _INPUTFILEREADER_HPP_

#include <string>
#include <memory>
#include <vector>

#include "toml.hpp"
#include "analysisRunner.hpp"

class InputFileReader
{
protected:
    std::string _inputFilename;
    toml::table _tomlTable;

    void parseTomlFile();

    size_t parseNumberOfAtomsPerMolecule();

    std::vector<std::string> parseXYZFiles();
    std::vector<size_t> parseAtomIndices();

public:
    explicit InputFileReader(const std::string_view &filename) : _inputFilename(filename) {}
    virtual ~InputFileReader() = default;

    virtual AnalysisRunner &read() = 0;
};

#endif