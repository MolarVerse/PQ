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

#include <algorithm>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "constants.hpp"
#include "exceptions.hpp"
#include "mathUtilities.hpp"
#include "orthorhombicBox.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "triclinicBox.hpp"
#include "units.hpp"

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
     * @param lineNumber the line number in the rst file being processed
     * @param settings the simulation settings (currently unused in this
     * function)
     * @param simulationBox the simulation box to which the box dimensions and
     * angles are applied
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
        size_t                   &lineNumber,
        Settings                 &settings,
        molsys::SimulationBox    &simulationBox
    )
    {
        // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        if ((lineElements.size() != 4) && (lineElements.size() != 7))
        {
            throw exc::RstFileException(
                std::format(
                    "Error in line {}: Box section must have 4 or 7 elements",
                    lineNumber
                )
            );
        }

        const auto boxDimensions = Pos3D{
            stod(lineElements[1]) * angstrom,
            stod(lineElements[2]) * angstrom,
            stod(lineElements[3]) * angstrom
        };
        // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

        auto checkPositive = [](auto dimension) { return dimension < 0.0; };

        if (std::ranges::any_of(boxDimensions, checkPositive))
            throw exc::RstFileException("All box dimensions must be positive");

        constexpr Angle defaultAngle = NINETY_DEG;
        auto boxAngles = Angle3D{defaultAngle, defaultAngle, defaultAngle};

        // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        if (7 == lineElements.size())
        {
            boxAngles = Angle3D{
                stod(lineElements[4]) * deg,
                stod(lineElements[5]) * deg,
                stod(lineElements[6]) * deg
            };

            auto checkAngles = [defaultAngle](Angle angle)
            { return angle < 0.0 || angle > 2.0 * defaultAngle; };

            if (std::ranges::any_of(boxAngles, checkAngles))
                throw exc::RstFileException(
                    "Box angles must be positive and smaller than 180°"
                );
        }
        // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

        const auto hasTriclinicAngles = !utilities::compare(
            boxAngles,
            Angle3D{defaultAngle, defaultAngle, defaultAngle},
            TRICLINIC_BOX_ANGLE_THRESHOLD
        );

        const auto &manostatSettings = settings.manostatSettings;
        const bool  isFullAnisotropic =
            manostatSettings.get_isotropy() == Isotropy::FULL_ANISOTROPIC;

        // A manostat with full anisotropic coupling shears the box, so it needs
        // a box that may change its angles, also if it starts with 90 degrees:
        // an orthorhombic box would take only the diagonal of the scaling while
        // the molecules are scaled with all of it.
        const bool needsShearableBox =
            isFullAnisotropic &&
            manostatSettings.get_manostatType() != ManostatType::NONE;

        if (hasTriclinicAngles || needsShearableBox)
        {
            const auto jobType = settings.generalSettings.getJobtype();

            // TODO(97gamjak): https://github.com/MolarVerse/PQ/issues/910
            if (jobType != JobType::QM_MD &&
                jobType != JobType::RING_POLYMER_QM_MD)
            {
                throw exc::InputFileException(
                    hasTriclinicAngles
                        ? "Triclinic box is only supported for QM-MD and "
                          "RP-QM-MD"
                        : "isotropy = full_anisotropic needs a triclinic box "
                          "(it changes the box angles), which is only "
                          "supported for QM-MD and RP-QM-MD"
                );
            }

            molsys::TriclinicBox box{
                boxDimensions,
                boxAngles,
                isFullAnisotropic
            };
            simulationBox.setBox(box, true);
        }
        else
        {
            molsys::OrthorhombicBox box(boxDimensions);
            simulationBox.setBox(box, true);
        }
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
