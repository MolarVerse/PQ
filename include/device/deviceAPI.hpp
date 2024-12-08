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

#ifndef __DEVICE_API_HPP__
#define __DEVICE_API_HPP__

#include <cstddef>   // for size_t

#include "deviceConfig.hpp"
#include "exceptions.hpp"

namespace device
{
    template <typename T>
    deviceError_t deviceFreeThrowError(T* ptr, const std::string& msg);

    template <typename T>
    deviceError_t deviceFree(T* ptr);

    template <typename T>
    deviceError_t deviceMemcpy(
        T*                       dst,
        const T*                 src,
        const size_t             size,
        const deviceMemcpyKind_t kind
    );

    template <typename T>
    deviceError_t deviceMemcpyAsync(
        T*                       dst,
        const T*                 src,
        const size_t             size,
        const deviceMemcpyKind_t kind,
        const deviceStream_t     stream
    );

}   // namespace device

#include "deviceAPI.tpp.hpp"   // DO NOT MOVE THIS LINE!!!

#endif __DEVICE_API_HPP__
