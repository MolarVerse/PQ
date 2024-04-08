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

#include "inputFileReader.hpp"

#include "tomlExtensions.hpp"

#include <bits/stdc++.h>
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
        cerr << "Error parsing file '" << *err.source().path << "':\n"
             << err.description() << "\n  (" << err.source().begin << ")\n";
        ::exit(-1);
    }
}

vector<string> InputFileReader::parseXYZFiles()
{
    const auto *input = _tomlTable["files"]["xyz"].as_array();

    return tomlExtensions::tomlArrayToVector<string>(input);
}

string InputFileReader::parseXYZOutputFile() { return _tomlTable["outputfiles"]["xyz"].value_or(""); }

vector<size_t> InputFileReader::parseAtomIndices()
{
    const auto *input = _tomlTable["system"]["atomIndices"].as_array();

    const auto atomIndicesToml = tomlExtensions::tomlArrayToVector<int64_t>(input);

    vector<size_t> atomIndices(atomIndicesToml.size());

    auto toSize_t = [](const int64_t &i) { return static_cast<int>(i); };

    std::ranges::copy(std::views::transform(atomIndicesToml, toSize_t), std::back_inserter(atomIndices));

    return atomIndices;
}

size_t InputFileReader::parseNumberOfAtomsPerMolecule()
{
    return static_cast<size_t>(_tomlTable["system"]["numberOfAtomsPerMolecule"].value_or(0));
}