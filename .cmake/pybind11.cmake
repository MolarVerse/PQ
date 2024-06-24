find_package(Python3 REQUIRED COMPONENTS Interpreter NumPy Development)

link_libraries(Python3::NumPy)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
)

FetchContent_MakeAvailable(pybind11)

if(NOT pybind11_FOUND)
    message(FATAL_ERROR "Failed to install the pybind11 python package")
else()
    message(STATUS "Successfully installed the pybind11 python package")
endif()

set(BUILD_WITH_PYBIND11 ON)
add_definitions(-DWITH_PYBIND11)