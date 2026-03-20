#version 460
//=========================================================
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
//=========================================================
#include "common.h"
//=========================================================
// size of a workgroup for compute
layout ( local_size_x = 16 ) in;
//=========================================================

/* Agent Update:
	SENSE - read the linearly filtered texture
	DECIDE - turn towards the highest concentration
	MOVE - you have a new direction, take a step that way
	DEPOSIT (now happens via raster)
*/
void main () {
	agents[ gl_GlobalInvocationID.x ].velocity += 0.1f;
}