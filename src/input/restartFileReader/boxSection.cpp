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

#include "boxSection.hpp"

#include <algorithm>     // for __any_of_fn, any_of
#include <format>        // for format
#include <functional>    // for identity
#include <string>        // for stod, string
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "engine.hpp"                  // for Engine
#include "exceptions.hpp"              // for RstFileException
#include "mathUtilities.hpp"           // for compare
#include "settings.hpp"                // for Settings
#include "simulationBox.hpp"           // for SimulationBox
#include "simulationBoxSettings.hpp"   // for SimulationBoxSettings
#include "vector3d.hpp"                // for Vec3D

using namespace input::restartFile;
using namespace customException;
using namespace linearAlgebra;
using namespace utilities;
using namespace settings;
using namespace simulationBox;
using namespace engine;

/**
 * @brief processes the box section of the rst file
 *
 * @details the box section can have 4 or 7 elements. If it has 4 elements, the
 * box is assumed to be orthogonal. If it has 7 elements, the box is assumed to
 * be triclinic. The second to fourth elements are the box dimensions, the next
 * 3 elements are the box angles.
 *
 * @param lineElements all elements of the line
 * @param engine object containing the engine
 *
 * @throws RstFileException if the number of elements in the
 * line is not 4 or 7
 * @throws RstFileException if the box dimensions are not
 * positive
 * @throws RstFileException if the box angles are not positive
 * or larger than 90°
 */
void BoxSection::process(
    std::vector<std::string> &lineElements,
    Engine           &engine
)
{
    if ((lineElements.size() != 4) && (lineElements.size() != 7))
        throw RstFileException(std::format(
            "Error in line {}: Box section must have 4 or 7 elements",
            _lineNumber
        ));

    const auto boxDimensions = Vec3D{
        stod(lineElements[1]),
        stod(lineElements[2]),
        stod(lineElements[3])
    };

    auto checkPositive = [](const double dimension) { return dimension < 0.0; };

    if (std::ranges::any_of(boxDimensions, checkPositive))
        throw RstFileException("All box dimensions must be positive");

    auto boxAngles = Vec3D{90.0, 90.0, 90.0};

    if (7 == lineElements.size())
    {
        boxAngles = Vec3D{
            stod(lineElements[4]),
            stod(lineElements[5]),
            stod(lineElements[6])
        };

        auto checkAngles = [](const double angle)
        { return angle < 0.0 || angle > 180.0; };

        if (std::ranges::any_of(boxAngles, checkAngles))
            throw RstFileException(
                "Box angles must be positive and smaller than 180°"
            );
    }

    if (!compare(boxAngles, Vec3D{90.0, 90.0, 90.0}, 1e-5))
    {
        auto box = TriclinicBox();
        box.setBoxAngles(boxAngles);
        box.setBoxDimensions(boxDimensions);
        engine.getSimulationBox().setBox(box);

        const auto jobType = Settings::getJobtype();

        // TODO: implement triclinic box for MM-MD
        if (jobType != JobType::QM_MD && jobType != JobType::RING_POLYMER_QM_MD)
            throw InputFileException(
                "Triclinic box is only supported for QM-MD and RP-QM-MD"
            );
    }
    else
    {
        auto box = OrthorhombicBox();
        box.setBoxDimensions(boxDimensions);
        engine.getSimulationBox().setBox(box);
    }

    SimulationBoxSettings::setBoxSet(true);
}

/**
 * @brief returns the keyword of the box section
 *
 * @return std::string
 */
std::string BoxSection::keyword() { return "box"; }

/**
 * @brief returns if the box section is a header
 *
 * @return true
 */
bool BoxSection::isHeader() { return true; }