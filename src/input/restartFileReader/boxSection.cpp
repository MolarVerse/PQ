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
#include <string>        // for stod, string
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "constants.hpp"
#include "engine.hpp"                  // for engine::Engine
#include "exceptions.hpp"              // for exc::RstFileException
#include "generalSettings.hpp"         // for Settings
#include "mathUtilities.hpp"           // for compare
#include "simulationBoxSettings.hpp"   // for SimulationBoxSettings
#include "triclinicBox.hpp"            // for TriclinicBox

namespace input::restartFile
{

    /**
     * @brief processes the box section of the rst file
     *
     * @details the box section can have 4 or 7 elements. If it has 4 elements,
     * the box is assumed to be orthogonal. If it has 7 elements, the box is
     * assumed to be triclinic. The second to fourth elements are the box
     * dimensions, the next 3 elements are the box angles.
     *
     * @param lineElements all elements of the line
     * @param engine object containing the engine
     *
     * @throws exc::RstFileException if the number of elements in the
     * line is not 4 or 7
     * @throws exc::RstFileException if the box dimensions are not
     * positive
     * @throws exc::RstFileException if the box angles are not positive
     * or larger than 90°
     */
    void BoxSection::process(
        std::vector<std::string> &lineElements,
        engine::Engine           &engine
    )
    {
        // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        if ((lineElements.size() != 4) && (lineElements.size() != 7))
        {
            throw exc::RstFileException(
                std::format(
                    "Error in line {}: Box section must have 4 or 7 elements",
                    _lineNumber
                )
            );
        }

        const auto boxDimensions = linalg::Vec3D{
            stod(lineElements[1]),
            stod(lineElements[2]),
            stod(lineElements[3])
        };
        // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

        auto checkPositive = [](const double dimension)
        { return dimension < 0.0; };

        if (std::ranges::any_of(boxDimensions, checkPositive))
            throw exc::RstFileException("All box dimensions must be positive");

        constexpr auto defaultAngle = 90.0;
        auto           boxAngles =
            linalg::Vec3D{defaultAngle, defaultAngle, defaultAngle};

        // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        if (7 == lineElements.size())
        {
            boxAngles = linalg::Vec3D{
                stod(lineElements[4]),
                stod(lineElements[5]),
                stod(lineElements[6])
            };

            auto checkAngles = [](const double angle)
            { return angle < 0.0 || angle > 2.0 * defaultAngle; };

            if (std::ranges::any_of(boxAngles, checkAngles))
                throw exc::RstFileException(
                    "Box angles must be positive and smaller than 180°"
                );
        }
        // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

        if (!utilities::compare(
                boxAngles,
                linalg::Vec3D{defaultAngle, defaultAngle, defaultAngle},
                TRICLINIC_BOX_ANGLE_THRESHOLD
            ))
        {
            molsys::TriclinicBox box;
            box.setBoxAngles(boxAngles);
            box.setBoxDimensions(boxDimensions);
            engine.getSimulationBox().setBox(box);

            const auto jobType = settings::GeneralSettings::getJobtype();

            // TODO: implement triclinic box for MM-MD
            if (jobType != JobType::QM_MD &&
                jobType != JobType::RING_POLYMER_QM_MD)
                throw exc::InputFileException(
                    "Triclinic box is only supported for QM-MD and RP-QM-MD"
                );
        }
        else
        {
            molsys::OrthorhombicBox box;
            box.setBoxDimensions(boxDimensions);
            engine.getSimulationBox().setBox(box);
        }

        settings::SimulationBoxSettings::setBoxSet(true);
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

}   // namespace input::restartFile
