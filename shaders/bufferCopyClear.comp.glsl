#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

layout ( local_size_x = 16, local_size_y = 16 ) in;

#include "common.h"

layout ( r32ui, set = 0, binding = 1 ) uniform uimage2D resolveBuffer;
layout ( r32f, set = 0, binding = 2 ) uniform image2D stateBuffer;

void main () {
	ivec2 pos = ivec2( gl_GlobalInvocationID.xy );
	if ( all( lessThan( gl_GlobalInvocationID.xy, imageSize( stateBuffer ) ) ) ) {
		float priorState = imageLoad( stateBuffer, pos ).r;
		float increment = float( imageLoad( resolveBuffer, pos ).r );

		// increment the state and clear the resolve buffer
		imageStore( stateBuffer, pos, vec4( priorState + increment ) );
		imageStore( resolveBuffer, pos, uvec4( 0 ) );
	}
}