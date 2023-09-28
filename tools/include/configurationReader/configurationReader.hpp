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

#ifndef _CONFIGURATION_READER_HPP_

#define _CONFIGURATION_READER_HPP_

#include "extxyzReader.hpp"   // for ExtxyzReader
#include "frame.hpp"          // for Frame
#include "vector3d.hpp"       // for fabs, Vec3D, Vector3D

#include <cstddef>   // for size_t
#include <fstream>   // for fstream
#include <string>    // for string
#include <vector>    // for vector

class ConfigurationReader
{
  private:
    std::vector<std::string> _filenames;

    std::fstream _fp;

    size_t               _nAtoms  = 0;
    size_t               _nFrames = 0;
    linearAlgebra::Vec3D _box;

    frameTools::Frame _frame;

    [[no_unique_address]] ExtxyzReader _extxyzReader;

  public:
    ConfigurationReader() = default;
    explicit ConfigurationReader(const std::vector<std::string> &filenames);

    bool               nextFrame();
    frameTools::Frame &getFrame();
    void               parseHeader();
    void               parseAtoms();
    [[nodiscard]] bool isBoxSet(const linearAlgebra::Vec3D &box) const { return fabs(box) > 1e-15; }
};

#endif   // _CONFIGURATION_READER_HPP_