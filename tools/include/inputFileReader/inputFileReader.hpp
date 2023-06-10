#ifndef _INPUTFILEREADER_HPP_

#define _INPUTFILEREADER_HPP_

#include <string>
#include <memory>

#include "toml.hpp"
#include "analysisRunner.hpp"

class InputFileReader
{
private:
    std::string _filename;
    toml::table _tomlTable;

    void parseXYZFILES();

public:
    explicit InputFileReader(const std::string_view &filename) : _filename(filename) {}
    virtual ~InputFileReader() = default;

    virtual AnalysisRunner &read() = 0;
};

#endif