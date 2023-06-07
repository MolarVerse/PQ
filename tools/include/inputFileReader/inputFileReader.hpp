#ifndef _INPUTFILEREADER_HPP_

#define _INPUTFILEREADER_HPP_

#include <string>

#include "toml.hpp"

class InputFileReader
{
private:
    std::string _filename;
    toml::table _tomlTable;

    void parseXYZFILES();

public:
    InputFileReader(const std::string_view &filename) : _filename(filename) {}
    virtual ~InputFileReader() = default;

    virtual void read();
};

#endif