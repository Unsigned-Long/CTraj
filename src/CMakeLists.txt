set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(Eigen3)
find_package(Sophus)
find_package(Ceres)
set(tiny-viewer_DIR  ${CMAKE_SOURCE_DIR}/thirdparty/tiny-viewer-install/lib/cmake/tiny-viewer)
find_package(tiny-viewer)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/core CORE_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/factor FACTOR_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/view VIEW_SRC_FILES)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/include/ctraj/core CORE_HEADER_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/include/ctraj/factor FACTOR_HEADER_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/include/ctraj/nofree NOFREE_HEADER_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/include/ctraj/spline SPLINE_HEADER_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/include/ctraj/utils UTILS_HEADER_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/include/ctraj/view VIEW_HEADER_FILES)

add_library(
        ${LIBRARY_NAME} SHARED
        ${CORE_SRC_FILES} ${FACTOR_SRC_FILES} ${VIEW_SRC_FILES}
        ${CORE_HEADER_FILES} ${FACTOR_HEADER_FILES} ${NOFREE_HEADER_FILES}
        ${SPLINE_HEADER_FILES} ${UTILS_HEADER_FILES} ${VIEW_HEADER_FILES}
)

set(ADDITIONAL_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

target_include_directories(
        ${LIBRARY_NAME} PUBLIC
        # only when building from the source tree
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # only when using the lib from the install path
        $<INSTALL_INTERFACE:include>
        ${ADDITIONAL_INCLUDE_DIRS}
)

target_link_libraries(
        ${LIBRARY_NAME} PUBLIC
        Sophus::Sophus
        ${CERES_LIBRARIES}
        tiny-viewer
)

# example & test

add_executable(${PROJECT_NAME}_prog ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(
        ${PROJECT_NAME}_prog PRIVATE
        ${LIBRARY_NAME}
)
