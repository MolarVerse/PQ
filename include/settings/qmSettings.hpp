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

#ifndef _QM_SETTINGS_HPP_

#define _QM_SETTINGS_HPP_

#include <cstddef>       // for size_t
#include <string>        // for string
#include <string_view>   // for string_view

namespace settings
{
    /**
     * @class enum QMMethod
     *
     */
    enum class QMMethod : size_t
    {
        NONE,
        DFTBPLUS,
        PYSCF,
        TURBOMOLE
    };

    std::string string(const QMMethod method);

    /**
     * @class QMSettings
     *
     * @brief stores all information about the external qm runner
     *
     */
    class QMSettings
    {
      private:
        static inline QMMethod    _qmMethod         = QMMethod::NONE;
        static inline std::string _qmScript         = "";
        static inline std::string _qmScriptFullPath = "";

      public:
        static void setQMMethod(const std::string_view &method);

        static void setQMMethod(const QMMethod method) { _qmMethod = method; }
        static void setQMScript(const std::string_view &script) { _qmScript = script; }
        static void setQMScriptFullPath(const std::string_view &script) { _qmScriptFullPath = script; }

        [[nodiscard]] static QMMethod    getQMMethod() { return _qmMethod; }
        [[nodiscard]] static std::string getQMScript() { return _qmScript; }
        [[nodiscard]] static std::string getQMScriptFullPath() { return _qmScriptFullPath; }
    };
}   // namespace settings

#endif   // _QM_SETTINGS_HPP_