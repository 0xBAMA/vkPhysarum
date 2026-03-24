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
float pheremone ( vec2 pos ) {
	// need to remap to texturespace... consider adding a jitter here as another agent parameter
	pos = pos / textureSize( state, 0 ).xy;
	return texture( state, pos.xy ).r;
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
		const vec2 avDir			= normalize( MYAGENT.velocity );
		const vec2 rightVec			= MYAGENT.senseDistance * rotate( avDir, -MYAGENT.senseAngle );
		const vec2 middleVec		= MYAGENT.senseDistance * avDir;
		const vec2 leftVec			= MYAGENT.senseDistance * rotate( avDir,  MYAGENT.senseAngle );
		const float rightSample		= pheremone( MYAGENT.position + rightVec );
		const float middleSample	= pheremone( MYAGENT.position + middleVec );
		const float leftSample		= pheremone( MYAGENT.position + leftVec );

		// make a decision on whether to turn left, right, go straight, or a random direction
		// this can be generalized and simplified, as some sort of weighted sum thing - will bear revisiting
		vec2 impulseVector = middleVec;
		if ( middleSample > leftSample && middleSample > rightSample ) {
			// just retain the existing direction
		} else if ( middleSample < leftSample && middleSample < rightSample ) { // turn a random direction
			impulseVector = RandomInUnitDisk();
		} else if ( rightSample > middleSample && middleSample > leftSample ) { // turn right (positive)
			impulseVector = rotate( middleVec, MYAGENT.turnAngle );
		} else if ( leftSample > middleSample && middleSample > rightSample ) { // turn left (negative)
			impulseVector = rotate( middleVec, -MYAGENT.turnAngle );
		}

		// apply impulse to an object of known mass + store back to SSBO
		vec2 acceleration = impulseVector * MYAGENT.forceAmount / MYAGENT.mass;	// get the resulting acceleration
		MYAGENT.velocity = MYAGENT.drag * MYAGENT.velocity + acceleration;					// compute the new velocity
		MYAGENT.position = wrap( MYAGENT.position + MYAGENT.velocity );				// get the new position

		// deposit
		imageAtomicAdd( resolveBuffer, ivec2( MYAGENT.position ), uint( MYAGENT.depositAmount ) );
	}
}