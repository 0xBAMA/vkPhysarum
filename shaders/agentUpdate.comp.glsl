#version 460
//=========================================================
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
//=========================================================
#include "common.h"
#include "random.h"
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

//=========================================================
#define WANGSEED PushConstants.wangSeed
#define MYAGENT agents[ gl_GlobalInvocationID.x ]
#define BUFFERSIZE vec2( GlobalData.floatBufferResolution )
//=========================================================
vec2 rotate ( const vec2 v, const float a ) {
	// takes argument in radians
	const float s = sin( a );
	const float c = cos( a );
	const mat2 m = mat2( c, -s, s, c );
	return m * v;
}
vec2 wrap ( vec2 pos ) {
	const ivec2 iS = ivec2( BUFFERSIZE );
	if ( pos.x >= iS.x ) pos.x -= iS.x;
	if ( pos.x < 0.0f ) pos.x += iS.x;
	if ( pos.y >= iS.y ) pos.y -= iS.y;
	if ( pos.y < 0.0f ) pos.y += iS.y;
	return pos;
}
//=========================================================
void main () {
	seed = WANGSEED + gl_GlobalInvocationID.x * 69420;

	if ( PushConstants.operation == -1 ) {
		// generating new, random values
		MYAGENT.position.x = BUFFERSIZE.x * NormalizedRandomFloat();
		MYAGENT.position.y = BUFFERSIZE.y * NormalizedRandomFloat();
	} else {
	// do the regular agent update...
		// sense taps, reading from the pheremone buffer

		// turn decision, based on the sense readings

		// move the agent based on the current velocity

		// wrap the position to keep it in-bounds for the raster process
		if ( clamp( MYAGENT.position, vec2( 0.0f ), BUFFERSIZE ) != MYAGENT.position ) {
			MYAGENT.position = wrap( MYAGENT.position );
		}
	}
}