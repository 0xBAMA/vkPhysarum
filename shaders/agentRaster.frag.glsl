#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#define RASTER 1
#include "common.h"

layout ( location = 0 ) out vec4 outFragColor;

void main () {
// this is for an additive raster process
//	float len = length( gl_PointCoord.xy - vec2( 0.5f ) );
//	if ( len < 0.5f )
//		outFragColor = vec4( 0.5f - len );
//	else
//		outFragColor = vec4( 0.0f );

	outFragColor = vec4( 1.0f );
}