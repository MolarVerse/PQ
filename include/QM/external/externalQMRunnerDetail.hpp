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

#ifndef QM__EXTERNAL__EXTERNAL_QM_RUNNER_DETAIL_HPP
#define QM__EXTERNAL__EXTERNAL_QM_RUNNER_DETAIL_HPP

#if !defined(_WIN32)
#include <sys/types.h>

#include <array>
#include <chrono>
#include <csignal>
#include <mstd/utils.hpp>

namespace QM::detail
{
    /**
     * @brief last SIGINT, SIGTERM or SIGHUP received while an external QM
     * program runs, 0 if none
     */
    struct CancelState
    {
        static inline volatile std::sig_atomic_t flag = 0;
    };

    /**
     * @brief records SIGINT, SIGTERM and SIGHUP while the QM process runs, so
     * its process group can be stopped before PQ exits; restores the previous
     * handlers on destruction
     */
    class ScopedCancelSignals
    {
       private:
        static constexpr auto _nSignals = 3;

        std::array<int, _nSignals> _signals{SIGINT, SIGTERM, SIGHUP};
        std::array<struct sigaction, _nSignals> _previous{};

       public:
        ScopedCancelSignals();
        ~ScopedCancelSignals();

        MSTD_NO_COPY(ScopedCancelSignals);
        MSTD_NO_MOVE(ScopedCancelSignals);
    };

    void stopProcessGroup(
        pid_t                     pid,
        int                      &status,
        std::chrono::milliseconds pollInterval
    );
}   // namespace QM::detail
#endif

#endif   // QM__EXTERNAL__EXTERNAL_QM_RUNNER_DETAIL_HPP
