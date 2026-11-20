add_library(privateHelper
    potentialSettingsUtils.cpp
)

target_include_directories(privateHelper
    PUBLIC
    ${PROJECT_SOURCE_DIR}/tests/include/privateHelper/
)

target_link_libraries(privateHelper
    PUBLIC
    settings
)
