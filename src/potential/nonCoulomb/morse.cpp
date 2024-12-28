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

#include "morse.hpp"

#include "settings.hpp"

using namespace potential;
using namespace settings;

/**
 * @brief Construct a new Morse object
 *
 * @param size
 */
Morse::Morse(const size_t size)
    : _size(size),
      _cutOffs(size * size),
      _params(size * size * _nParams * _nParams)
{
}

/**
 * @brief Construct a new Morse object
 *
 */
Morse::Morse() { *this = Morse(0); }

/**
 * @brief Resize the Morse object
 *
 * @param size
 */
void Morse::resize(const size_t size)
{
    _params.resize(size * size * _nParams * _nParams);
    _cutOffs.resize(size * size);
    _size = size;
}

/**
 * @brief Add a pair of Morse types to the Morse object
 *
 * @param pair
 * @param index1
 * @param index2
 */
void Morse::addPair(
    const MorsePair& pair,
    const size_t     index1,
    const size_t     index2
)
{
    const auto newIndex1   = (index1 * _size + index2) * _nParams;
    const auto newIndex2   = (index2 * _size + index1) * _nParams;
    _params[newIndex1]     = pair.getDissociationEnergy();
    _params[newIndex2]     = pair.getDissociationEnergy();
    _params[newIndex1 + 1] = pair.getWellWidth();
    _params[newIndex2 + 1] = pair.getWellWidth();
    _params[newIndex1 + 2] = pair.getEquilibriumDistance();
    _params[newIndex2 + 2] = pair.getEquilibriumDistance();
    _params[newIndex1 + 3] = pair.getEnergyCutOff();
    _params[newIndex2 + 3] = pair.getEnergyCutOff();
    _params[newIndex1 + 4] = pair.getForceCutOff();
    _params[newIndex2 + 4] = pair.getForceCutOff();

    _cutOffs[index1 * _size + index2] = pair.getRadialCutOff();
    _cutOffs[index2 * _size + index1] = pair.getRadialCutOff();
}

/**
 * @brief Get the parameters of all pairs of Morse types
 *
 * @return std::vector<Real>
 */
std::vector<Real> Morse::copyParams() const { return _params; }

/**
 * @brief Get the cutoffs of all pairs of Morse types
 *
 * @return std::vector<Real>
 */
std::vector<Real> Morse::copyCutOffs() const { return _cutOffs; }

/**
 * @brief Get the number of parameters
 *
 * @return size_t
 */
size_t Morse::getNParams() { return _nParams; }

/**
 * @brief Get the size of the Morse object
 *
 * @return size_t
 */
size_t Morse::getSize() const { return _size; }