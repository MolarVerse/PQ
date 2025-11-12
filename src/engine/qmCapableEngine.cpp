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

#include "qmCapableEngine.hpp"

using namespace engine;
using namespace settings;
using namespace QM;

using std::shared_ptr;

/**
 * @brief Set the QM runner based on the specified method
 *
 * @param method The QM method to use
 */
void QMCapableEngine::setQMRunner(const QMMethod method)
{
    _qmRunner = QMRunnerManager::createQMRunner(method);
}

/**
 * @brief Get the QM runner
 *
 * @return shared_ptr<QMRunner> Shared pointer to the QM runner
 */
shared_ptr<QMRunner> QMCapableEngine::getQMRunner() const { return _qmRunner; }

/**
 * @brief Check if QM runner is set
 *
 * @return bool True if QM runner is set, false otherwise
 */
bool QMCapableEngine::hasQMRunner() const { return _qmRunner != nullptr; }
