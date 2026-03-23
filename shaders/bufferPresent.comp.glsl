#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

layout ( local_size_x = 16, local_size_y = 16 ) in;

// the draw image
layout ( rgba16f, set = 0, binding = 0 ) uniform image2D image;

// the state image
layout ( set = 0, binding = 1 ) uniform sampler2D state;


void main () {
	vec2 loc = ( gl_GlobalInvocationID.xy + vec2( 0.5f ) ) / ;
	imageStore( image, ivec2( gl_GlobalInvocationID.xy ), texture( state,  ) );
}