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

#ifndef UTILITIES__TERMINAL_PROGRESS_BAR_HPP
#define UTILITIES__TERMINAL_PROGRESS_BAR_HPP

#include <mstd/utils.hpp>
#include <optional>
#include <ostream>

#include "progressbar.hpp"

namespace utilities
{
    [[nodiscard]] bool isStdoutTerminal();

    /**
     * @brief progress bar that writes nothing unless the output is a terminal
     */
    class TerminalProgressBar
    {
       private:
        std::optional<progressbar> _bar;

       public:
        explicit TerminalProgressBar(int nSteps);
        TerminalProgressBar(int nSteps, std::ostream &ostream, bool isTerminal);
        ~TerminalProgressBar() = default;

        MSTD_NO_COPY(TerminalProgressBar);
        MSTD_NO_MOVE(TerminalProgressBar);

        void update();
    };

}   // namespace utilities

#endif   // UTILITIES__TERMINAL_PROGRESS_BAR_HPP
