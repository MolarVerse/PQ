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

#ifndef _SETUP_HPP_

#define _SETUP_HPP_

#include <string>   // for string

#include "typeAliases.hpp"

/**
 * @namespace setup
 *
 * @note
 *  This namespace contains all the functions that are used to setup the
 *  simulation. This includes reading the input file, the moldescriptor,
 *  the rst file, the guff.dat file, and post processing the setup.
 *
 */
namespace setup
{
    void setupRequestedJob(const std::string &inputFileName, pq::Engine &);

    void startSetup(pq::Timer &, pq::Timer &, pq::Engine &);
    void endSetup(const pq::Timer &, pq::Timer &, pq::Engine &);

    void readFiles(pq::Engine &);
    void setupEngine(pq::Engine &);

#ifdef __PQ_GPU__
    void initDeviceMemory();
#endif
}   // namespace setup

#endif   // _SETUP_HPP_