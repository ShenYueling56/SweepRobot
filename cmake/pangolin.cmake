add_definitions(-DUSE_PANGOLIN)
find_package(Pangolin REQUIRED)
include_directories(
        ${Pangolin_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
        ${Pangolin_LIBRARIES}
        )