add_library(pq_python
    selection.cpp
)

target_include_directories(pq_python
    PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include/python
)

target_link_libraries(pq_python
    PUBLIC
    Python::Python
    Python::NumPy
)

install(TARGETS pq_python
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)