#include <metal_stdlib>
using namespace metal;

struct DilationParams {
    float4x4 proj;
    uint2 texSize;
    float voxDist;
    float voxSize;
    int stepSize;
};

constant int2 kOffsets[8] = {
    int2(1, 0), int2(1, 1), int2(0, 1), int2(-1, 1),
    int2(-1, 0), int2(-1, -1), int2(0, -1), int2(1, -1)
};

kernel void initDepthDilation(
    texture2d<float, access::read> depthTex [[texture(0)]],
    texture2d<float, access::write> dest [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= depthTex.get_width() || gid.y >= depthTex.get_height()) return;

    float depth = depthTex.read(gid).r;

    float4 val = float4(float2(gid), depth, 0.0);
    dest.write(val, gid);
}

kernel void dilateDepthStep(
    texture2d<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> dest [[texture(1)]],
    constant DilationParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.texSize.x || gid.y >= params.texSize.y) return;

    float4 val = src.read(gid);
    float focalLength = params.proj[0][0];

    for (int i = 0; i < 8; i++) {
        uint2 srcCoord = uint2(int2(gid) + kOffsets[i] * params.stepSize);

        bool withinBounds = srcCoord.x < params.texSize.x && srcCoord.y < params.texSize.y;
        if (!withinBounds) continue;

        float4 srcVal = src.read(srcCoord);

        // Convert NDC depth to linear
        float A = params.proj[2][2];
        float B = params.proj[2][3];
        float srcZ = srcVal.z * 2.0 - 1.0;
        float srcDepthLinear = abs(B / (srcZ + A));

        float pixelSize = 2.0 * srcDepthLinear / (focalLength * float(params.texSize.x));
        float radius = (params.voxDist + params.voxSize) / pixelSize;

        float2 diff = srcVal.xy - float2(gid);
        bool withinRadius = dot(diff, diff) < radius * radius;
        bool newMinimum = srcVal.z < val.z;
        bool notZero = srcVal.z != 0.0;

        if (withinRadius && newMinimum && notZero) {
            val = srcVal;
        }
    }

    dest.write(val, gid);
}
