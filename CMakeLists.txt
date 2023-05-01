cmake_minimum_required(VERSION 3.7...3.26)
include(.cmake/documentation.cmake)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(PIMD-QMCF VERSION 1.0
                  DESCRIPTION "Path Integral Molecular Dynamics - Quantum Charge Field"
                  LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
    message("Release build.")
    set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -Werror -pedantic -O3")
else()
    message("Debug build.")
    set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -g")
endif()

add_library(setup           
    SHARED 
        src/setup/rstFileReader.cpp 
        include/setup/rstFileReader.hpp 
        src/setup/rstFileSection.cpp 
        include/setup/rstFileSection.hpp
)
add_library(simulationBox   
    SHARED 
        src/simulationBox/simulationBox.cpp 
        include/simulationBox/simulationBox.hpp 
        src/simulationBox/box.cpp 
        include/simulationBox/box.hpp
)
add_library(settings        
    SHARED 
        src/settings/settings.cpp           
        include/settings/settings.hpp
)
add_library(stringUtilities 
    SHARED 
        src/utilities/stringUtilities.cpp   
        include/utilities/stringUtilities.hpp
)

target_include_directories(setup           
    PUBLIC 
        include/setup 
        include/simulationBox 
        include/settings 
        include/utilities
)

target_include_directories(simulationBox
    PUBLIC 
        include/simulationBox
)
target_include_directories(settings        
    PUBLIC 
        include/settings
)
target_include_directories(stringUtilities 
    PUBLIC 
        include/utilities
)

add_executable(pimd_qmcf src/pimd_qmcf.cpp)
target_link_libraries(pimd_qmcf 
    PUBLIC 
        setup 
        simulationBox 
        settings 
        stringUtilities
)

BUILD_DOCUMENTATION()
