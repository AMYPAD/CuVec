cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
if("${CUVEC_BUILD_VERSION}" STREQUAL "")
  set(CUVEC_BUILD_VERSION 2 CACHE STRING "version" FORCE)
endif()
project(cuvec LANGUAGES C CXX VERSION "${CUVEC_BUILD_VERSION}")

option(CUVEC_CUDA_OPTIONAL "Make CUDA optional rather than forced" ON)

cmake_policy(PUSH)
cmake_policy(SET CMP0074 NEW)  # <PackageName>_ROOT hints for find_package
cmake_policy(SET CMP0104 NEW)  # CMAKE_CUDA_ARCHITECTURES
find_package(Python3 COMPONENTS Interpreter Development REQUIRED) # NumPy
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
  find_package(PythonExtensions REQUIRED)
  set(LIB_TYPE "MODULE")

  find_package(SWIG 3.0 COMPONENTS python)
  if(SWIG_FOUND)
  include(${SWIG_USE_FILE})
  set(${CMAKE_PROJECT_NAME}_SWIG_SRC "${CMAKE_CURRENT_LIST_DIR}/include/${CMAKE_PROJECT_NAME}.i")
  set_source_files_properties("${${CMAKE_PROJECT_NAME}_SWIG_SRC}" PROPERTIES CPLUSPLUS ON)
  endif()
else()
  set(LIB_TYPE "SHARED")
endif()
cmake_policy(POP)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
option(CUVEC_DEBUG "Print out CUDA malloc & free operations" OFF)
if(CUVEC_DEBUG)
  add_compile_definitions(CUVEC_DEBUG)
endif(CUVEC_DEBUG)
message(STATUS "CuVec debugging: ${CUVEC_DEBUG}")

set(${CMAKE_PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include/")  # / suffix important
install(DIRECTORY "${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}" DESTINATION ${CMAKE_PROJECT_NAME}/include)

# main project

file(GLOB SRC LIST_DIRECTORIES false "src/*.cu")
include_directories(src)
include_directories(${Python3_INCLUDE_DIRS})
#include_directories(${Python3_NumPy_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED ${SRC})
add_library(AMYPAD::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}>"
  "$<INSTALL_INTERFACE:${CMAKE_PROJECT_NAME}/include>")
if(CUDAToolkit_FOUND)
  target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES} CUDA::cudart_static)
else()
  set_source_files_properties(${SRC} PROPERTIES LANGUAGE CXX)
  target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})
endif()

if(SKBUILD)
  python_extension_module(${PROJECT_NAME})
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 11
  #VERSION ${CMAKE_PROJECT_VERSION} SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
  PREFIX ""  # remove shared lib prefix to make importable
  INTERFACE_${PROJECT_NAME}_MAJOR_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${PROJECT_NAME}_MAJOR_VERSION)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
  INCLUDES DESTINATION ${CMAKE_PROJECT_NAME}/include
  LIBRARY DESTINATION ${CMAKE_PROJECT_NAME})
install(EXPORT ${PROJECT_NAME}Targets FILE AMYPAD${PROJECT_NAME}Targets.cmake
  NAMESPACE AMYPAD:: DESTINATION ${CMAKE_PROJECT_NAME}/cmake)

# alternative swvec module
if(SWIG_FOUND AND SKBUILD)
  if(CUDAToolkit_FOUND)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    set_source_files_properties(src/swvec.i PROPERTIES COMPILE_DEFINITIONS _CUVEC_HALF=__half)
  endif()
  set_source_files_properties(src/swvec.i PROPERTIES CPLUSPLUS ON)
  set_source_files_properties(src/swvec.i PROPERTIES USE_TARGET_INCLUDE_DIRECTORIES ON)
  swig_add_library(swvec LANGUAGE python SOURCES src/swvec.i)
  if(CUDAToolkit_FOUND)
    swig_link_libraries(swvec CUDA::cudart_static)
  endif()
  target_include_directories(swvec PUBLIC
    "$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}>"
    "$<INSTALL_INTERFACE:${CMAKE_PROJECT_NAME}/include>")
  python_extension_module(swvec)
  set_target_properties(swvec PROPERTIES
    CXX_STANDARD 11
    VERSION ${CMAKE_PROJECT_VERSION} SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
    INTERFACE_swvec_MAJOR_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})
  set_property(TARGET swvec APPEND PROPERTY COMPATIBLE_INTERFACE_STRING swvec_MAJOR_VERSION)
  install(TARGETS swvec
    INCLUDES DESTINATION ${CMAKE_PROJECT_NAME}/include
    LIBRARY DESTINATION ${CMAKE_PROJECT_NAME})
endif()

add_subdirectory(src/example_mod)
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
