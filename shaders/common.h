// trying this out - keeping a common set of descriptors across all shaders

//push constants block
layout( push_constant ) uniform constants {
	mat4 render_matrix;
	float tOffset;
	// VertexBuffer ;
} PushConstants;

// first global config etc data
layout( set = 0, binding = 0 ) uniform GlobalData {
	mat4 placeholder0;
	mat4 placeholder1;
	mat4 placeholder2;
	vec4 placeholder3;
	vec4 placeholder4;
	vec4 placeholder5;
	vec4 placeholder6;
} globalData;

// then the SSBO for the agents
struct Agent {
	float mass;
	float pad;
	float drag;
	float senseDistance;
	float senseAngle;
	float turnAngle;
	float forceAmount; // replaces stepsize
	float depositAmount;
	vec2 position;
	vec2 velocity;
};

layout ( buffer_reference, std430 ) buffer AgentBuffer {
	Agent agents[];
};

// then is the two float buffers...
	// eventually I'd like to go to a sampler array and get this fully bindless

