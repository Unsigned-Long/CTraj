cmake_minimum_required(VERSION 3.10)

project(ctraj VERSION 1.0)

# -----------
# set lib name ans space
# -----------
set(LIBRARY_NAME ctraj)

set(LIBRARY_NAMESPACE CTraj)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -Wno-unused-function")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

# -----------
# for install
# -----------

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Export package for use from the build tree
set(LIBRARY_CMAKE_EXPORT_DIR ${CMAKE_INSTALL_DATADIR}/${LIBRARY_NAME}/cmake)

set_target_properties(${LIBRARY_NAME} PROPERTIES EXPORT_NAME ${LIBRARY_NAMESPACE})

export(
        TARGETS ${LIBRARY_NAME}
        NAMESPACE ${LIBRARY_NAMESPACE}::
        FILE ${LIBRARY_NAMESPACE}Targets.cmake
)
export(
        PACKAGE ${LIBRARY_NAMESPACE}
)

configure_package_config_file(
        ${LIBRARY_NAMESPACE}Config.cmake.in
        ${CMAKE_BINARY_DIR}/${LIBRARY_NAMESPACE}Config.cmake
        INSTALL_DESTINATION ${LIBRARY_CMAKE_EXPORT_DIR}
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Write version to file
write_basic_package_version_file(
        ${LIBRARY_NAMESPACE}ConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)
