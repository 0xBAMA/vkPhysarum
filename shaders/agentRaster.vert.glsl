#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "common.h"

void main () {

	// placeholder, can potentially do larger and do some kernel work in the frag shader
	gl_PointSize = 1.0f;

}