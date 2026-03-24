//=========================================================
// push constants block - updated at smallest scope
layout( push_constant ) uniform constants {
// RNG seeding from the CPU
	uint wangSeed;

// specifying specific operations to be performed
	// e.g. if I want to randomly seed the agent positions
	int operation;

} PushConstants;

//=========================================================
// Global config etc data in a UBO
layout( set = 0, binding = 0 ) uniform globalData {
	// buffer resolutions:
	uvec2 floatBufferResolution;
	uvec2 presentBufferResolution;

	// some initial usage here for base parameters + jitter
		// this is used to specify small variation on a single "preset"

	/* some other parameterization lives here for when we want to reinit, like:

		blur radius
		decay rate
		...

	*/

	// seeding the deterministic wang hash for each Agent
	uint AgentGenSeed;
	float AgentGenSpread;

	// diffuse/decay parameterization
	float decayRate;
	float radius;

	// visualizer config
	float brightnessScale;

} GlobalData;
//=========================================================

#ifndef saturate
#define saturate(x) clamp(x, 0, 1)
#endif

#ifndef UINT_MAX
#define UINT_MAX (0xFFFFFFFF-1)
#endif

#ifndef PI_DEFINED
#define PI_DEFINED
const float pi = 3.141592f;
const float tau = 2.0f * pi;
const float sqrtpi = 1.7724538509f;
#endif