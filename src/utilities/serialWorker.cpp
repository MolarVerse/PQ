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

#include "serialWorker.hpp"

#include <iostream>
#include <utility>

namespace utilities
{
    /**
     * @brief start the background thread
     *
     * @param capacity maximum number of queued and running tasks, at least 1
     */
    SerialWorker::SerialWorker(const std::size_t capacity)
        : _capacity(capacity > 0 ? capacity : 1), _thread([this] { _work(); })
    {
    }

    /**
     * @brief finish all queued tasks and join the thread
     *
     * @details a destructor must not throw, so an error nobody rethrew is
     * printed to stderr instead of being lost
     */
    SerialWorker::~SerialWorker()
    {
        {
            const std::scoped_lock lock(_mutex);
            _stopping = true;
        }

        _changed.notify_all();
        _thread.join();

        if (!_error)
            return;

        try
        {
            std::rethrow_exception(_error);
        }
        catch (const std::exception &error)
        {
            std::cerr << "Error in background output: " << error.what() << '\n';
        }
        catch (...)
        {
            std::cerr << "Unknown error in background output\n";
        }
    }

    /**
     * @brief queue a task, blocking while capacity tasks are in flight
     *
     * @param task
     *
     * @throw the first exception of an earlier task
     */
    void SerialWorker::submit(Task task)
    {
        {
            std::unique_lock lock(_mutex);

            _changed.wait(
                lock,
                [this]
                {
                    const auto inFlight = _tasks.size() + (_running ? 1 : 0);
                    return _error || inFlight < _capacity;
                }
            );

            _rethrowError();
            _tasks.push_back(std::move(task));
        }

        _changed.notify_all();
    }

    /**
     * @brief block until all queued tasks are done
     *
     * @throw the first exception of an earlier task
     */
    void SerialWorker::wait()
    {
        std::unique_lock lock(_mutex);

        _changed.wait(lock, [this] { return _tasks.empty() && !_running; });

        _rethrowError();
    }

    /**
     * @brief rethrow and clear a stored task error; call with _mutex held
     */
    void SerialWorker::_rethrowError()
    {
        if (_error)
            std::rethrow_exception(std::exchange(_error, nullptr));
    }

    /**
     * @brief thread loop: run tasks in order until stopped and empty
     */
    void SerialWorker::_work()
    {
        while (true)
        {
            Task task;

            {
                std::unique_lock lock(_mutex);

                _changed.wait(
                    lock,
                    [this] { return _stopping || !_tasks.empty(); }
                );

                if (_tasks.empty())
                    return;

                task = std::move(_tasks.front());
                _tasks.pop_front();
                _running = true;
            }

            std::exception_ptr error;

            try
            {
                task();
            }
            catch (...)
            {
                error = std::current_exception();
            }

            {
                const std::scoped_lock lock(_mutex);

                if (error && !_error)
                    _error = error;

                _running = false;
            }

            _changed.notify_all();
        }
    }

}   // namespace utilities
