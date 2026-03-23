// trying this out - keeping a common set of descriptors across all shaders
//=========================================================
// push constants block - updated at smallest scope
layout( push_constant ) uniform constants {
// RNG seeding from the CPU
	uint wangSeed;

// specifying specific operations to be performed
	// e.g. if I want to randomly seed the agent positions
	int operation;

// specifying the blur operation
	int blurOperation;

} PushConstants;
//=========================================================
// Global config etc data in a UBO
layout( set = 0, binding = 0 ) uniform globalData {
	// buffer resolutions:
	uvec2 floatBufferResolution;
	uvec2 presentBufferResolution;

	// some initial usage here for base parameters + jitter
		// this is used to specify small variation on a single "preset"

	/* some other parameterization lives here, like:

		blur radius
		decay rate
		...

	*/


} GlobalData;
//=========================================================
// then is the two float image buffers...
	// eventually I'd like to go to a sampler array and get this fully bindless...
//=========================================================
// I need:
// linear filtered access to Float Buffer A
layout ( set = 0, binding = 2 ) uniform sampler2D FloatBufferA_tex;
// image load/store access to Float Buffer A
layout ( r32f, set = 0, binding = 3 ) uniform image2D FloatBufferA_img;
// image load/store access to Float Buffer B
layout ( r32f, set = 0, binding = 4 ) uniform image2D FloatBufferB_img;
//=========================================================