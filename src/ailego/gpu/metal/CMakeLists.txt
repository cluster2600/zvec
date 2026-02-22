#
#  CMakeLists.txt for Metal GPU module
#

# Only build on Apple platforms
if(NOT APPLE)
    return()
endif()

# Check for Metal support
set(METAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/zvec_metal.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/zvec_metal.metal
)

# Create Metal library
add_library(zvec_metal STATIC
    ${METAL_SOURCES}
)

# Metal compilation flags
set_target_properties(zvec_metal PROPERTIES
    COMPILE_FLAGS "-fvisibility=hidden"
    LINK_FLAGS "-framework Metal -framework MetalKit"
)

# Include directories
target_include_directories(zvec_metal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install
install(TARGETS zvec_metal
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
