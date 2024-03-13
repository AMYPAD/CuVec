cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
if(NOT DEFINED SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION 6 CACHE STRING "version" FORCE)
endif()
string(REGEX REPLACE [[([0-9]+)\.([0-9]+)\.([0-9]+).*]] [[\1.\2.\3]] SKBUILD_PROJECT_VERSION "${SKBUILD_PROJECT_VERSION}")
project(cuvec LANGUAGES C CXX VERSION "${SKBUILD_PROJECT_VERSION}")

option(CUVEC_CUDA_OPTIONAL "Make CUDA optional rather than forced" ON)

cmake_policy(PUSH)
cmake_policy(SET CMP0074 NEW)  # <PackageName>_ROOT hints for find_package
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA arch" FORCE)
endif()
cmake_policy(SET CMP0104 NEW)  # CMAKE_CUDA_ARCHITECTURES
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
if(NOT CUVEC_CUDA_OPTIONAL)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
else()
  find_package(CUDAToolkit)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
  else()
    message(WARNING "Could not find CUDA: using CPU")
    add_compile_definitions(CUVEC_DISABLE_CUDA)
    #list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      add_definitions("-x c++")
    endif()
  endif()
endif()
if(SKBUILD)
  find_package(pybind11 CONFIG)
  find_package(SWIG 4.0)
  if(SWIG_FOUND)
    include(${SWIG_USE_FILE})
    set(${CMAKE_PROJECT_NAME}_SWIG_SRC "${CMAKE_CURRENT_LIST_DIR}/include/${CMAKE_PROJECT_NAME}.i")
    set_source_files_properties("${${CMAKE_PROJECT_NAME}_SWIG_SRC}" PROPERTIES CPLUSPLUS ON)
  endif()
endif()
cmake_policy(POP)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(CUVEC_DEBUG "Print out CUDA malloc & free operations" OFF)
if(CUVEC_DEBUG)
  add_compile_definitions(CUVEC_DEBUG)
  message(STATUS "CuVec debugging: TRUE")
else()
  message(STATUS "CuVec debugging: FALSE")
endif(CUVEC_DEBUG)

set(${CMAKE_PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include/")  # / suffix important
install(DIRECTORY "${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}" DESTINATION ${CMAKE_PROJECT_NAME}/include)

# cpython extension

file(GLOB SRC LIST_DIRECTORIES false "src/cpython*.cu")
include_directories(src)
include_directories(${Python_INCLUDE_DIRS})

if(SKBUILD)
  python_add_library(${PROJECT_NAME}_cpython MODULE WITH_SOABI ${SRC})
else()
  add_library(${PROJECT_NAME}_cpython SHARED ${SRC})
endif()
add_library(AMYPAD::${PROJECT_NAME}_cpython ALIAS ${PROJECT_NAME}_cpython)
target_include_directories(${PROJECT_NAME}_cpython PUBLIC
  "$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}>"
  "$<INSTALL_INTERFACE:${CMAKE_PROJECT_NAME}/include>")
if(CUDAToolkit_FOUND)
  target_link_libraries(${PROJECT_NAME}_cpython PRIVATE CUDA::cudart_static)
else()
  set_source_files_properties(${SRC} PROPERTIES LANGUAGE CXX)
  target_link_libraries(${PROJECT_NAME}_cpython PRIVATE)
endif()

set_target_properties(${PROJECT_NAME}_cpython PROPERTIES
  CXX_STANDARD 11
  VERSION ${CMAKE_PROJECT_VERSION} SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
  INTERFACE_${PROJECT_NAME}_cpython_MAJOR_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})
set_property(TARGET ${PROJECT_NAME}_cpython APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${PROJECT_NAME}_cpython_MAJOR_VERSION)
install(TARGETS ${PROJECT_NAME}_cpython EXPORT ${PROJECT_NAME}Targets
  INCLUDES DESTINATION ${CMAKE_PROJECT_NAME}/include
  LIBRARY DESTINATION ${CMAKE_PROJECT_NAME})
install(EXPORT ${PROJECT_NAME}Targets FILE AMYPAD${PROJECT_NAME}Targets.cmake
  NAMESPACE AMYPAD:: DESTINATION ${CMAKE_PROJECT_NAME}/cmake)

# alternative pybind11 extension

if(pybind11_FOUND AND SKBUILD)
  file(GLOB SRC LIST_DIRECTORIES false "src/pybind11*.cu")
  # include_directories(src)
  # include_directories(${Python_INCLUDE_DIRS})
  pybind11_add_module(${PROJECT_NAME}_pybind11 MODULE WITH_SOABI ${SRC})
  add_library(AMYPAD::${PROJECT_NAME}_pybind11 ALIAS ${PROJECT_NAME}_pybind11)
  target_include_directories(${PROJECT_NAME}_pybind11 PUBLIC
    "$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}>"
    "$<INSTALL_INTERFACE:${CMAKE_PROJECT_NAME}/include>")
  if(CUDAToolkit_FOUND)
    target_link_libraries(${PROJECT_NAME}_pybind11 PRIVATE CUDA::cudart_static)
  else()
    set_source_files_properties(${SRC} PROPERTIES LANGUAGE CXX)
  endif()

  set_target_properties(${PROJECT_NAME}_pybind11 PROPERTIES
    CXX_STANDARD 11
    VERSION ${CMAKE_PROJECT_VERSION} SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
    INTERFACE_pybind11_MAJOR_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})
  set_property(TARGET ${PROJECT_NAME}_pybind11 APPEND PROPERTY COMPATIBLE_INTERFACE_STRING pybind11_MAJOR_VERSION)
  install(TARGETS ${PROJECT_NAME}_pybind11
    INCLUDES DESTINATION ${CMAKE_PROJECT_NAME}/include
    LIBRARY DESTINATION ${CMAKE_PROJECT_NAME})
endif()

# alternative swig extension

if(SWIG_FOUND AND SKBUILD)
  if(CUDAToolkit_FOUND)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    set_source_files_properties(src/cuvec_swig.i PROPERTIES COMPILE_DEFINITIONS _CUVEC_HALF=__half)
  endif()
  set_source_files_properties(src/cuvec_swig.i PROPERTIES CPLUSPLUS ON)
  set_source_files_properties(src/cuvec_swig.i PROPERTIES USE_TARGET_INCLUDE_DIRECTORIES ON)
  swig_add_library(${PROJECT_NAME}_swig LANGUAGE python
    OUTPUT_DIR "${SKBUILD_PLATLIB_DIR}/${CMAKE_PROJECT_NAME}"
    SOURCES src/cuvec_swig.i)
  if(WIN32)
    set_property(TARGET ${PROJECT_NAME}_swig PROPERTY SUFFIX ".${Python_SOABI}.pyd")
  else()
    set_property(TARGET ${PROJECT_NAME}_swig PROPERTY SUFFIX ".${Python_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
  endif()
  if(CUDAToolkit_FOUND)
    target_link_libraries(${PROJECT_NAME}_swig PRIVATE Python::Module CUDA::cudart_static)
  else()
    target_link_libraries(${PROJECT_NAME}_swig PRIVATE Python::Module)
  endif()
  target_include_directories(${PROJECT_NAME}_swig PUBLIC
    "$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}>"
    "$<INSTALL_INTERFACE:${CMAKE_PROJECT_NAME}/include>")
  set_target_properties(${PROJECT_NAME}_swig PROPERTIES
    CXX_STANDARD 11
    VERSION ${CMAKE_PROJECT_VERSION} SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
    INTERFACE_${PROJECT_NAME}_swig_MAJOR_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})
  set_property(TARGET ${PROJECT_NAME}_swig APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${PROJECT_NAME}_swig_MAJOR_VERSION)
  install(TARGETS ${PROJECT_NAME}_swig
    INCLUDES DESTINATION ${CMAKE_PROJECT_NAME}/include
    LIBRARY DESTINATION ${CMAKE_PROJECT_NAME})
endif()

# example projects

add_subdirectory(src/example_cpython)
if(pybind11_FOUND AND SKBUILD)
  add_subdirectory(src/example_pybind11)
endif()
add_subdirectory(src/example_swig)

# install project

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/AMYPAD${CMAKE_PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_PROJECT_NAME}/cmake)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/AMYPAD${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
  VERSION "${CMAKE_PROJECT_VERSION}" COMPATIBILITY AnyNewerVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/AMYPAD${CMAKE_PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/AMYPAD${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_PROJECT_NAME}/cmake)
