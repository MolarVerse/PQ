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

#include "boxSection.hpp"

#include "engine.hpp"                  // for Engine
#include "exceptions.hpp"              // for RstFileException
#include "mathUtilities.hpp"           // for compare
#include "simulationBox.hpp"           // for SimulationBox
#include "simulationBoxSettings.hpp"   // for SimulationBoxSettings
#include "vector3d.hpp"                // for Vec3D

#include <algorithm>     // for __any_of_fn, any_of
#include <format>        // for format
#include <functional>    // for identity
#include <string>        // for stod, string
#include <string_view>   // for string_view
#include <vector>        // for vector

using namespace input::restartFile;

/**
 * @brief processes the box section of the rst file
 *
 * @details the box section can have 4 or 7 elements. If it has 4 elements, the box is assumed to be orthogonal. If it has 7
 * elements, the box is assumed to be triclinic. The second to fourth elements are the box dimensions, the next 3 elements are the
 * box angles.
 *
 * @param lineElements all elements of the line
 * @param engine object containing the engine
 *
 * @throws customException::RstFileException if the number of elements in the line is not 4 or 7
 * @throws customException::RstFileException if the box dimensions are not positive
 * @throws customException::RstFileException if the box angles are not positive or larger than 90°
 */
void BoxSection::process(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    if ((lineElements.size() != 4) && (lineElements.size() != 7))
        throw customException::RstFileException(
            std::format("Error in line {}: Box section must have 4 or 7 elements", _lineNumber));

    const auto boxDimensions = linearAlgebra::Vec3D{stod(lineElements[1]), stod(lineElements[2]), stod(lineElements[3])};

    if (std::ranges::any_of(boxDimensions, [](double dimension) { return dimension < 0.0; }))
        throw customException::RstFileException("All box dimensions must be positive");

    auto boxAngles = linearAlgebra::Vec3D{90.0, 90.0, 90.0};

    if (7 == lineElements.size())
    {
        boxAngles = linearAlgebra::Vec3D{stod(lineElements[4]), stod(lineElements[5]), stod(lineElements[6])};

        if (std::ranges::any_of(boxAngles, [](double angle) { return angle < 0.0 || angle > 180.0; }))
            throw customException::RstFileException("Box angles must be positive and smaller than 180°");
    }

    if (!utilities::compare(boxAngles, linearAlgebra::Vec3D{90.0, 90.0, 90.0}, 1e-5))
    {
        auto box = simulationBox::TriclinicBox();
        box.setBoxAngles(boxAngles);
        box.setBoxDimensions(boxDimensions);
        engine.getSimulationBox().setBox(box);
    }
    else
    {
        auto box = simulationBox::OrthorhombicBox();
        box.setBoxDimensions(boxDimensions);
        engine.getSimulationBox().setBox(box);
    }

    settings::SimulationBoxSettings::setBoxSet(true);
}