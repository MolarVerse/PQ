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

#ifndef UTILITIES__SERIAL_WORKER_HPP
#define UTILITIES__SERIAL_WORKER_HPP

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace utilities
{
    /**
     * @brief runs tasks one after another on a single background thread
     *
     * @details at most capacity tasks are queued or running; submit blocks
     * until there is room. The first exception a task throws is rethrown on
     * the caller's thread by the next submit or wait. The destructor finishes
     * all queued tasks.
     */
    class SerialWorker
    {
       public:
        using Task = std::move_only_function<void()>;

       private:
        std::size_t             _capacity;
        std::mutex              _mutex;
        std::condition_variable _changed;
        std::deque<Task>        _tasks;
        bool                    _running  = false;
        bool                    _stopping = false;
        std::exception_ptr      _error;

        // last member: started after and joined before the state above
        std::jthread _thread;

       public:
        explicit SerialWorker(std::size_t capacity);
        ~SerialWorker();

        SerialWorker(const SerialWorker &)            = delete;
        SerialWorker &operator=(const SerialWorker &) = delete;
        SerialWorker(SerialWorker &&)                 = delete;
        SerialWorker &operator=(SerialWorker &&)      = delete;

        void submit(Task task);
        void wait();

       private:
        void _work();
        void _rethrowError();
    };

}   // namespace utilities

#endif   // UTILITIES__SERIAL_WORKER_HPP
