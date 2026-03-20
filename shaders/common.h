// trying this out - keeping a common set of descriptors across all shaders
//=========================================================
// push constants block
layout( push_constant ) uniform constants {
// buffer resolutions:
	uvec2 floatBufferResolution;
	uvec2 presentBufferResolution;

// RNG seeding
	uint wangSeed;

// specifying specific operations to be performed
	// e.g. if I want to randomly seed the agent positions
	uint operation;

// can also include the ranges for the Agent generation parameters
	// we have plenty of space (using 24 of 256 bytes so far)

} PushConstants;
//=========================================================
// Global config etc data in a UBO
layout( set = 0, binding = 0 ) uniform GlobalData {
	mat4 placeholder0;
	mat4 placeholder1;
	mat4 placeholder2;
	vec4 placeholder3;
	vec4 placeholder4;
	vec4 placeholder5;
	vec4 placeholder6;
} globalData;
//=========================================================
// then the SSBO for the agents
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