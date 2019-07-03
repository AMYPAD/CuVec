set(${CMAKE_PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include/")  # / suffix important
install(DIRECTORY "${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}" DESTINATION niftypet/${CMAKE_PROJECT_NAME}/include)

add_subdirectory(prc)
