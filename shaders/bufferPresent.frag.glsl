#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "common.h"

layout ( location = 0 ) out vec4 outFragColor;

void main () {
	// need to take a texture sample from Float Buffer A
	 vec2 sampleLocation = vec2( 0.0f );
//	sampleLocation = ( gl_FragCoord.xy + 0.5f ) / PushConstants.presentBufferResolution;
	outFragColor = vec4( sampleLocation, sin( globalData.placeholder6.x * 0.01f ), 1.0f );
}