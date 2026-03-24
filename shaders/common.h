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

	float decayRate;
	float radius;

} GlobalData;
//=========================================================