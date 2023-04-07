set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(Eigen3)
find_package(Sophus)
find_package(Ceres)
find_package(SlamSceneViewer)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/core CORE_SRC_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/view VIEW_SRC_FILES)

add_library(
        ${LIBRARY_NAME} SHARED
        ${CORE_SRC_FILES} ${VIEW_SRC_FILES}
)

target_include_directories(
        ${LIBRARY_NAME} PUBLIC
        ${EIGEN_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        # for interface
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include>
)

target_link_libraries(
        ${LIBRARY_NAME} PUBLIC
        Sophus::Sophus
        ${CERES_LIBRARIES}
        SlamSceneViewer::SlamSceneViewer
)

# example & test

add_executable(${PROJECT_NAME}_prog ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(
        ${PROJECT_NAME}_prog PRIVATE
        ${LIBRARY_NAME}
)
