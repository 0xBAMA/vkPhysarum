#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "common.h"

layout ( location = 0 ) out vec4 outFragColor;

void main () {
	// need to take a texture sample from Float Buffer A
	 vec2 sampleLocation = vec2( 0.0f );
	sampleLocation = ( gl_FragCoord.xy + 0.5f ) / GlobalData.presentBufferResolution;

	outFragColor = vec4( texture( FloatBufferA_tex, sampleLocation ).xyz, 1.0f );
//	outFragColor = vec4( sampleLocation, 0.0f, 1.0f );
//	outFragColor = vec4( imageLoad( FloatBufferA_img, ivec2( gl_FragCoord.xy ) ).xyz / 10.0f, 1.0f );
}