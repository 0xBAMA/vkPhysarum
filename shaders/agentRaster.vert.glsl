#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "common.h"

void main () {

	// placeholder, can potentially do larger and do some kernel work in the frag shader
	gl_PointSize = 1.0f;

	// writing the point locations
	vec2 pos = agents[ gl_VertexIndex ].position / GlobalData.floatBufferResolution;
	gl_Position = vec4( 2.0f * ( pos - 0.5f ), 0.0f, 1.0f );
}