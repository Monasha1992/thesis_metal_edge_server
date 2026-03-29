#include <metal_stdlib>
using namespace metal;

#define EMPTY_VOXEL -1.0
#define PLAYER_TOP 0.2
#define PLAYER_BOTTOM 1.7
#define PLAYER_RADIUS 0.5
#define MIN_DOT 0.3

struct IntegrateParams {
    float4x4 view;
    float4x4 proj;
    float4x4 viewInv;
    float4x4 projInv;
    
    uint3 voxCount;
    float voxSize;
    float voxDist;
    float voxMin;
    float depthDispThresh;
    int numPlayers;
};

// --- Helper functions ---

float3 eyePos(float4x4 viewInvMat) {
    return float3(viewInvMat[0][3], viewInvMat[1][3], viewInvMat[2][3]);
}

float4 worldToHCS(float3 worldPos, float4x4 viewMat, float4x4 projMat) {
    return projMat * (viewMat * float4(worldPos, 1.0));
}

float3 hcsToNDC(float4 hcs) {
    return (hcs.xyz / hcs.w) * 0.5 + 0.5;
}

float3 worldToNDC(float3 worldPos, float4x4 viewMat, float4x4 projMat) {
    return hcsToNDC(worldToHCS(worldPos, viewMat, projMat));
}

float3 ndcToWorld(float3 ndc, float4x4 viewInvMat, float4x4 projInvMat) {
    float4 hcs = float4(ndc * 2.0 - 1.0, 1.0);
    float4 worldH = viewInvMat * (projInvMat * hcs);
    return worldH.xyz / worldH.w;
}

float ndcToLinear(float depthNDC, float4x4 projMat) {
    float z = depthNDC * 2.0 - 1.0;
    float A = projMat[2][2];
    float B = projMat[2][3];
    return abs(B / (z + A));
}

float3 voxelToWorld(uint3 indices, uint3 voxCount, float voxSize) {
    float3 pos = float3(indices) + 0.5;
    pos -= float3(voxCount) / 2.0;
    pos *= voxSize;
    return pos;
}

uint3 worldToVoxel(float3 pos, uint3 voxCount, float voxSize) {
    pos /= voxSize;
    pos += float3(voxCount) / 2.0;
    uint3 id = uint3(floor(pos));
    id = clamp(id, uint3(0), voxCount);
    return id;
}

// --- Kernels ---

kernel void clearVolume(
    texture3d<float, access::write> volume [[texture(0)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= volume.get_width() || gid.y >= volume.get_height() || gid.z >= volume.get_depth()) return;
    volume.write(float4(EMPTY_VOXEL, 0, 0, 0), gid);
}

kernel void integrate(
    device float3* frustumVolume [[buffer(0)]],
    constant IntegrateParams& params [[buffer(1)]],
    texture2d<float, access::read> depthTex [[texture(0)]],
    texture2d<float, access::read> normTex [[texture(1)]],
    texture2d<float, access::read> dilatedDepth [[texture(2)]],
    texture3d<float, access::read_write> volume [[texture(3)]],
    uint id [[thread_position_in_grid]]
) {
    float3 vLocalPos = frustumVolume[id];
    float3 vWorldPos = (params.viewInv * float4(vLocalPos, 1.0)).xyz;
    uint3 coord = worldToVoxel(vWorldPos, params.voxCount, params.voxSize);
    float3 voxPos = voxelToWorld(coord, params.voxCount, params.voxSize);

    float3 eye = eyePos(params.viewInv);
    float3 eyeToVox = voxPos - eye;
    float voxEyeDist = length(eyeToVox);

    float3 voxNDC = worldToNDC(voxPos, params.view, params.proj);

    // Sample depth and normal
    uint2 depthCoord = uint2(voxNDC.xy * float2(depthTex.get_width(), depthTex.get_height()));
    depthCoord = clamp(depthCoord, uint2(0), uint2(depthTex.get_width()-1, depthTex.get_height()-1));
    
    float depthNDC = depthTex.read(depthCoord).r;
    float3 depthNorm = normTex.read(depthCoord).xyz;
    float3 depthPos = ndcToWorld(float3(voxNDC.xy, depthNDC), params.viewInv, params.projInv);

    float3 eyeToDepth = depthPos - eye;
    float depthEyeDist = length(eyeToDepth);
    float normDot = -dot(eyeToVox, depthNorm) / voxEyeDist;

    float sDist = depthEyeDist - voxEyeDist;
    sDist *= saturate(normDot);

    bool empty = sDist >= 1.0;
    float sDistNorm = min(sDist / params.voxDist, 1.0);
    bool withinBand = sDistNorm >= -params.voxMin / params.voxSize;

    // Dilation test
    float dilatedDepthNDC = dilatedDepth.read(depthCoord).z;
    float dilatedDepthLinear = ndcToLinear(dilatedDepthNDC, params.proj);
    float depthLinear = ndcToLinear(depthNDC, params.proj);
    float3 voxView = (params.view * float4(voxPos, 1.0)).xyz;
    bool acceptableDepthDisparity = depthLinear < dilatedDepthLinear + params.depthDispThresh;
    bool unoccludedByDilation = abs(voxView.z) < dilatedDepthLinear || acceptableDepthDisparity;
    bool validSurfaceNormal = empty || normDot > MIN_DOT;

    bool valid = withinBand && validSurfaceNormal && unoccludedByDilation;
    if (valid) {
        volume.write(float4(sDistNorm, 0, 0, 0), coord);
    }
}

struct VoxelData {
    uint coordX;
    uint coordY;
    uint coordZ;
    float value;
};

kernel void extractNonEmpty(
    texture3d<float, access::read> volume [[texture(0)]],
    device atomic_uint* counter [[buffer(0)]],
    device VoxelData* output [[buffer(1)]],
    constant uint& maxOutput [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= volume.get_width() || gid.y >= volume.get_height() || gid.z >= volume.get_depth()) return;

    float val = volume.read(gid).r;
    if (val == EMPTY_VOXEL) return;

    uint idx = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
    if (idx >= maxOutput) return;

    output[idx].coordX = gid.x;
    output[idx].coordY = gid.y;
    output[idx].coordZ = gid.z;
    output[idx].value = val;
}

