add_library(QM
    qmRunner.cpp
    dftbplusRunner.cpp
)

target_include_directories(QM
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/QM
)

target_link_libraries(QM
    PUBLIC
    simulationBox
)

install(TARGETS QM
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)