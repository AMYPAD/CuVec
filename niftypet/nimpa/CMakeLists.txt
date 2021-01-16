set(${CMAKE_PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include/")  # / suffix important
install(DIRECTORY "${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}" DESTINATION niftypet/${CMAKE_PROJECT_NAME}/include)

project(cuvec)

file(GLOB SRC LIST_DIRECTORIES false "src/*.cu")
include_directories(src)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED ${SRC})
add_library(NiftyPET::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}>"
  "$<INSTALL_INTERFACE:niftypet/${CMAKE_PROJECT_NAME}/include>")
target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES} ${CUDA_LIBRARIES})

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
  INCLUDES DESTINATION niftypet/${CMAKE_PROJECT_NAME}/include
  LIBRARY DESTINATION niftypet/${CMAKE_PROJECT_NAME})
install(EXPORT ${PROJECT_NAME}Targets FILE NiftyPET${PROJECT_NAME}Targets.cmake
  NAMESPACE NiftyPET:: DESTINATION niftypet/${CMAKE_PROJECT_NAME}/cmake)

add_subdirectory(prc)
