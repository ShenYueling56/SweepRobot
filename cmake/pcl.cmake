find_package(PCL REQUIRED)
include_directories(
        ${PCL_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
        ${PCL_LIBRARIES}
        )