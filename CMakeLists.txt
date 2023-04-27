cmake_minimum_required(VERSION 3.7...3.26)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(MyProject VERSION 1.0
                  DESCRIPTION "Path Integral Molecular Dynamics - Quantum Charge Field"
                  LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Werror -pedantic -g")

add_library(rstFileReader   SHARED src/rstFileReader/rstFileReader.cpp include/rstFileReader/rstFileReader.hpp)
add_library(simulationBox   SHARED src/simulationBox/simulationBox.cpp include/simulationBox/simulationBox.hpp)
add_library(settings        SHARED src/settings/settings.cpp           include/settings/settings.hpp)
add_library(stringUtilities SHARED src/utilities/stringUtilities.cpp   include/utilities/stringUtilities.hpp)

target_include_directories(rstFileReader   PUBLIC include/rstFileReader include/simulationBox include/settings include/utilities)
target_include_directories(simulationBox   PUBLIC include/simulationBox)
target_include_directories(settings        PUBLIC include/settings)
target_include_directories(stringUtilities PUBLIC include/utilities)

add_executable(pimd_qmcf src/pimd_qmcf.cpp)
target_link_libraries(pimd_qmcf PUBLIC rstFileReader simulationBox settings stringUtilities)


