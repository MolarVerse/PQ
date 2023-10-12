find_package(Python COMPONENTS Interpreter Development)

if(Python_FOUND AND Python_VERSION_MAJOR EQUAL 3)
    add_custom_target(python_tools COMMAND ${Python_EXECUTABLE} -m pip install --break-system-packages ${_pip_args} ${CMAKE_CURRENT_SOURCE_DIR})
endif()
