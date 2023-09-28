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

#ifndef _TOOLENGINE_HPP_

#define _TOOLENGINE_HPP_

#include "inputFileReader.hpp"   // for InputFileReader

#include <functional>    // for function
#include <map>           // for map
#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

class AnalysisRunner;   // forward declaration

class Engine
{
  private:
    std::string _executableName;
    std::string _inputFilename;

    std::map<std::string, std::function<void(const std::string_view &)>> _analysisRunnerKeys;

    std::vector<InputFileReader *> _inputFileReaders;
    std::vector<AnalysisRunner *>  _analysisRunners;

  public:
    Engine(const std::string_view &, const std::string_view &);
    ~Engine()
    {
        for (auto inputFileReader : _inputFileReaders)
            delete inputFileReader;
    }

    void run();
    void addAnalysisRunnerKeys();
    void parseAnalysisRunners();
};

#endif   // _ENGINE_HPP_