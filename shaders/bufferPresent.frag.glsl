#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "common.h"

layout ( location = 0 ) out vec4 outFragColor;

void main () {
	// need to take a texture sample from Float Buffer A
	vec2 sampleLocation = ( gl_FragCoord.xy + 0.5f ) / PushConstants.presentBufferResolution;

}