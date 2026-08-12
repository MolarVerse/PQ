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

#ifndef _SINGLETON_HPP_
#define _SINGLETON_HPP_

/**
 * @brief Singleton class template.
 *
 * This header file defines a Singleton class template that provides a
 * thread-safe implementation of the Singleton design pattern. It ensures that
 * only one instance of the specified type T is created and provides global
 * access to that instance.
 *
 * @tparam T The type of the singleton instance.
 */
template <typename T>
class Singleton
{
   public:
    static T& getInstance()
    {
        static T instance;
        return instance;
    }

    Singleton(const Singleton&)            = delete;
    Singleton& operator=(const Singleton&) = delete;

   protected:
    Singleton()  = default;
    ~Singleton() = default;
};

#endif   // _SINGLETON_HPP_
