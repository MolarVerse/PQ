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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "serialWorker.hpp"
#include "throwWithMessage.hpp"

TEST(TestSerialWorker, runsTasksInOrder)
{
    std::vector<int> order;
    std::vector<int> expected;

    {
        utilities::SerialWorker worker{2};
        for (int i = 0; i < 10; ++i)
        {
            worker.submit([&order, i] { order.push_back(i); });
            expected.push_back(i);
        }
        worker.wait();
    }

    EXPECT_EQ(order, expected);
}

TEST(TestSerialWorker, destructorFinishesQueuedTasks)
{
    std::atomic_int done{0};

    {
        utilities::SerialWorker worker{2};
        for (int i = 0; i < 5; ++i) worker.submit([&done] { ++done; });
    }

    EXPECT_EQ(done, 5);
}

TEST(TestSerialWorker, waitRethrowsTaskErrorOnce)
{
    utilities::SerialWorker worker{2};
    worker.submit([] { throw std::runtime_error("disk full"); });

    EXPECT_THROW_MSG(worker.wait(), std::runtime_error, "disk full");
    EXPECT_NO_THROW(worker.wait());
}

TEST(TestSerialWorker, submitRethrowsEarlierTaskError)
{
    utilities::SerialWorker worker{1};
    worker.submit([] { throw std::runtime_error("disk full"); });

    // capacity 1: the next submit waits for the failed task, then throws
    EXPECT_THROW_MSG(worker.submit([] {}), std::runtime_error, "disk full");
}

TEST(TestSerialWorker, submitBlocksAtCapacity)
{
    utilities::SerialWorker worker{1};
    std::atomic_bool        release{false};
    std::atomic_bool        secondQueued{false};

    worker.submit(
        [&release]
        {
            while (!release) std::this_thread::yield();
        }
    );

    std::jthread producer{[&]
                          {
                              worker.submit([] {});
                              secondQueued = true;
                          }};

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(secondQueued);

    release = true;
    producer.join();
    EXPECT_TRUE(secondQueued);
}
