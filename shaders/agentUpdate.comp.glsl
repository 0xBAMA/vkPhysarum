#version 460
//=========================================================
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
//=========================================================
#include "common.h"
#include "random.h"
//=========================================================
// size of a workgroup for compute
layout ( local_size_x = 256 ) in;
//=========================================================

/* Agent Update:
	SENSE - read the linearly filtered texture
	DECIDE - turn towards the highest concentration
	MOVE - you have a new direction, take a step that way
	DEPOSIT - imageAtomicAdd
*/

//=========================================================
// then the SSBO for the agents (only accessed by this shader)
struct Agent {
// simulation parameters
	float mass;
	float pad;
	float drag;
	float senseDistance;
	float senseAngle;
	float turnAngle;
	float forceAmount; // replaces stepsize
	float depositAmount;

// dynamic sim state
	vec2 position;
	vec2 velocity;
};
//=========================================================
layout ( set = 0, binding = 1, std430 ) buffer AgentBuffer {
	Agent agents[];
};

layout ( set = 0, binding = 2 ) uniform sampler2D state;

layout ( r32ui, set = 0, binding = 3 ) uniform uimage2D resolveBuffer;
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

	if ( PushConstants.operation != 0 ) {
		// generating new, random values from the UBO seed
		uint seedCache = seed;
		seed = PushConstants.operation;

		// calculate the deterministic starting values
		MYAGENT.mass = mix( 1.5f, 30.0f, NormalizedRandomFloat() );
		MYAGENT.drag = mix( 0.5f, 1.0f, NormalizedRandomFloat() );
		MYAGENT.senseDistance = mix( 5.0f, 20.0f, NormalizedRandomFloat() );
		MYAGENT.senseAngle = mix( 0.0f, tau, NormalizedRandomFloat() );
		MYAGENT.turnAngle = mix( 0.0f, tau, NormalizedRandomFloat() );
		MYAGENT.forceAmount = mix( 0.1f, 2.0f, NormalizedRandomFloat() );
		MYAGENT.depositAmount = mix( 10.0f, 1000.0f, NormalizedRandomFloat() );

		// these things are not deterministic
		seed = seedCache;
		MYAGENT.position.x = BUFFERSIZE.x * NormalizedRandomFloat();
		MYAGENT.position.y = BUFFERSIZE.y * NormalizedRandomFloat();
		MYAGENT.velocity = 0.1f * normalize( RandomInUnitDisk() );

	} else {
	// do the regular agent update...
		// sense taps, reading from the pheremone buffer

//		float senseTaps[ 3 ] = float[ 3 ](
//			texture( state, ).r,
//		);

		// turn decision, based on the sense readings

		// move the agent based on the current velocity
		MYAGENT.position += MYAGENT.velocity;
	}

	// wrap the position to keep it in-bounds for the raster process
	if ( clamp( MYAGENT.position, vec2( 0.0f ), BUFFERSIZE ) != MYAGENT.position ) {
		MYAGENT.position = wrap( MYAGENT.position );
	}

	// need to tally the contribution for this update
	imageAtomicAdd( resolveBuffer, ivec2( MYAGENT.position ), uint( 100 + MYAGENT.depositAmount ) );
}