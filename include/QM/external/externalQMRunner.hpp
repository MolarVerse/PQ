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

#ifndef _EXTERNAL_QM_RUNNER_HPP_

#define _EXTERNAL_QM_RUNNER_HPP_

#include "qmRunner.hpp"
#include "typeAliases.hpp"

namespace QM
{
    /**
     * @brief ExternalQMRunner inherits from QMRunner
     *
     */
    class ExternalQMRunner : public QMRunner
    {
       protected:
        std::string       _scriptPath  = SCRIPT_PATH_;
        const std::string _singularity = SINGULARITY_;
        const std::string _staticBuild = STATIC_BUILD_;

       public:
        ExternalQMRunner()           = default;
        ~ExternalQMRunner() override = default;

        void         run(pq::SimBox &, pq::PhysicalData &) override;
        virtual void execute() = 0;

        virtual void writeCoordsFile(pq::SimBox &) = 0;
        virtual void readStressTensor(pq::Box &, pq::PhysicalData &) {}

        void readForceFile(pq::SimBox &, pq::PhysicalData &);
        void readChargeFile(pq::SimBox &);

        /*******************************
         * standard getter and setters *
         *******************************/

        [[nodiscard]] const std::string &getScriptPath() const;
        [[nodiscard]] const std::string &getSingularity() const;
        [[nodiscard]] const std::string &getStaticBuild() const;

        void setScriptPath(const std::string_view &scriptPath);
    };
}   // namespace QM

#endif   // _EXTERNAL_QM_RUNNER_HPP_