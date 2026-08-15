#ifndef _DRIVER_HPP_
#define _DRIVER_HPP_

#include <string>

namespace driver
{
    /**
     * @brief The Driver class is responsible for running a PQ simulation from
     * an input file.
     */
    class Driver
    {
       public:
        void run(const std::string &inputFileName);
    };
}   // namespace driver

#endif   // _DRIVER_HPP_
