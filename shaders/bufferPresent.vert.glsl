#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "common.h"

void main () {
	// need explicit vertex positions, because we are doing simple indexed draw
	switch( gl_VertexIndex ) {
		case 0: gl_Position = vec4( -1.0f, -1.0f, 0.0f, 1.0f ); break;
		case 1: gl_Position = vec4(  3.0f, -1.0f, 0.0f, 1.0f ); break;
		case 2: gl_Position = vec4( -1.0f,  3.0f, 0.0f, 1.0f ); break;
	}
}