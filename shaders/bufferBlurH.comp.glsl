#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

layout ( local_size_x = 16, local_size_y = 16 ) in;

#include "common.h"

layout ( set = 0, binding = 1 ) uniform sampler2D stateBuffer;
layout ( r32f, set = 0, binding = 2 ) uniform image2D scratchBuffer;

float blurWeight ( const float pos ) {
	// divide by number of taps included here instead of in the loop... proportional to the filter radius (ignores a scale factor... still ok?)
	float normalizationTerm = GlobalData.radius * sqrtpi; // integral of distribution is sqrt(pi/a), we precompute this constant
	float gaussianWeight = exp( -( pos * pos ) / ( 2.0f * GlobalData.radius * GlobalData.radius ) );
	return gaussianWeight / normalizationTerm;
}

float blurResult ( vec2 uv ) {
	float val = 0.0f;

	// 3 * radius based on observation that it's within some reasonable threshold of zero by that point (3 stdev)
	// also, 99.8% of the integral is inside of 3 standard deviations
	for ( float offset = -3.0f * GlobalData.radius - 0.5f; offset < 3.0f * GlobalData.radius + 0.5f; offset++ )
	// https://www.shadertoy.com/view/Xd33Rf note use of texel border sampling to double effective bandwidth
		val += blurWeight( offset ) * texture( stateBuffer, ( uv + ( vec2( offset, 0.0f ) / textureSize( stateBuffer, 0 ).xy ) ) ).r;

	return val;
}

void main () {
	vec2 p = vec2( gl_GlobalInvocationID.xy + 0.5f ) / imageSize( scratchBuffer ).xy;
	imageStore( scratchBuffer, ivec2( gl_GlobalInvocationID.xy ), vec4( blurResult( p ) ) );
}