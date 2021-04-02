cmake_minimum_required(VERSION 3.9)

project(trimesh)

# == Compile option to fix static/dynamic library linking
add_compile_options(-fPIC)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# Eigen
set(EIGEN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${EIGEN_INCLUDE_DIR})

# IGL
set(LIBIGL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/libigl)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"                 ON)
option(LIBIGL_WITH_PREDICATES        "Use exact predicates"         ON)
#set(LIBIGL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${LIBIGL_INCLUDE_DIR})
find_package(LIBIGL REQUIRED)

#add_subdirectory(PNG)
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

if(MSVC)
# JPEG
    add_subdirectory(JPEG)
    set(JPEG_LIBRARIES JPEG)
else(MSVC)
    find_package(JPEG REQUIRED)
    include_directories(${JPEG_INCLUDE_DIR})
endif(MSVC)





# == External
add_subdirectory(include/geometry-central)
add_subdirectory(include/pybind11)
include_directories(include/libigl/include)
include_directories(include/libigl/include/igl/predicates)
include_directories(include/libigl)





# == Build our project stuff
pybind11_add_module(trimesh src/main.cpp)
target_include_directories(trimesh PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(trimesh PRIVATE geometry-central igl::core igl::predicates pybind11::embed ${PNG_LIBRARY} ${JPEG_LIBRARIES})
