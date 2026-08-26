add_library(virial
    virial.cpp
)

target_include_directories(virial
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/virial
)

target_link_libraries(virial
    PRIVATE
    molsys
)

install(TARGETS virial
    DESTINATION lib
)
