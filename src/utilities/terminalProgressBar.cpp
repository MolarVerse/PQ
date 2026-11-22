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

#include "terminalProgressBar.hpp"

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <cstdio>
#include <iostream>

namespace utilities
{
    /**
     * @brief check if stdout is a terminal
     *
     * @return true if stdout is a terminal, false if it is redirected
     */
    bool isStdoutTerminal()
    {
#ifdef _WIN32
        return ::_isatty(::_fileno(stdout)) != 0;
#else
        return ::isatty(STDOUT_FILENO) != 0;
#endif
    }

    /**
     * @brief progress bar on stdout, shown only if stdout is a terminal
     *
     * @param nSteps number of steps
     */
    TerminalProgressBar::TerminalProgressBar(int nSteps)
        : TerminalProgressBar(nSteps, std::cout, isStdoutTerminal())
    {
    }

    /**
     * @brief progress bar on ostream, shown only if isTerminal is true
     *
     * @param nSteps number of steps
     * @param ostream output stream of the bar
     * @param isTerminal true if ostream is a terminal
     */
    TerminalProgressBar::TerminalProgressBar(
        int           nSteps,
        std::ostream &ostream,
        bool          isTerminal
    )
    {
        if (isTerminal)
        {
            _bar.emplace(nSteps, true, ostream);
        }
    }

    /**
     * @brief advance the bar by one step
     */
    void TerminalProgressBar::update()
    {
        if (_bar.has_value())
        {
            _bar->update();
        }
    }

}   // namespace utilities
