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

#include "lennardJones.hpp"

#include "settings.hpp"

using namespace potential;
using namespace settings;

/**
 * @brief Construct a new Lennard Jones object
 *
 * @param size
 */
LennardJones::LennardJones(const size_t size)
    : _size(size),
      _cutOffs(size * size),
      _params(size * size * _nParams * _nParams)
{
}

/**
 * @brief Construct a new Lennard Jones object
 *
 */
LennardJones::LennardJones() { *this = LennardJones(0); }

/**
 * @brief Resize the LennardJones object
 *
 * @param size
 */
void LennardJones::resize(const size_t size)
{
    _params.resize(size * size * _nParams * _nParams);
    _cutOffs.resize(size * size);
    _size = size;
}

/**
 * @brief Add a pair of Lennard-Jones types to the LennardJones object
 *
 * @param pair
 * @param index1
 * @param index2
 */
void LennardJones::addPair(
    const LennardJonesPair& pair,
    const size_t            index1,
    const size_t            index2
)
{
    const auto newIndex1   = (index1 * _size + index2) * _nParams;
    const auto newIndex2   = (index2 * _size + index1) * _nParams;
    _params[newIndex1]     = pair.getC6();
    _params[newIndex2]     = pair.getC6();
    _params[newIndex1 + 1] = pair.getC12();
    _params[newIndex2 + 1] = pair.getC12();
    _params[newIndex1 + 2] = pair.getEnergyCutOff();
    _params[newIndex2 + 2] = pair.getEnergyCutOff();
    _params[newIndex1 + 3] = pair.getForceCutOff();
    _params[newIndex2 + 3] = pair.getForceCutOff();

    _cutOffs[index1 * _size + index2] = pair.getRadialCutOff();
    _cutOffs[index2 * _size + index1] = pair.getRadialCutOff();
}

/**
 * @brief Get the parameters of all pairs of Lennard-Jones types
 *
 * @return std::vector<Real>
 */
std::vector<Real> LennardJones::copyParams() const { return _params; }

/**
 * @brief Get the cutoffs of all pairs of Lennard-Jones types
 *
 * @return std::vector<Real>
 */
std::vector<Real> LennardJones::copyCutOffs() const { return _cutOffs; }

/**
 * @brief Get the number of parameters
 *
 * @return size_t
 */
size_t LennardJones::getNParams() { return _nParams; }

/**
 * @brief Get the size of the LennardJones object
 *
 * @return size_t
 */
size_t LennardJones::getSize() const { return _size; }