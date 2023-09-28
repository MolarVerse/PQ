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

    std::string parseXYZOutputFile();
    std::vector<std::string> parseXYZFiles();
    std::vector<size_t> parseAtomIndices();

public:
    explicit InputFileReader(const std::string_view &filename) : _inputFilename(filename) {}
    virtual ~InputFileReader() = default;

    virtual AnalysisRunner &read() = 0;
};

#endif