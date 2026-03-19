#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "common.h"

//=========================================================
// size of a workgroup for compute
layout ( local_size_x = 16, local_size_y = 16 ) in;
// descriptor bindings for the pipeline

void main () {

}