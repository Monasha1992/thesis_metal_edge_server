// Metal Shading Language source compiled at runtime via MTLDevice.makeLibrary(source:).
//
// Ported from:
//   DepthKit.hlsl         → depth coordinate helpers
//   EnvMapper.hlsl        → voxel ↔ world helpers
//   EnvironmentMapping.compute  → Clear, InitDepthDilation, DilateDepthStep, Integrate
//   DepthNorm.compute     → DepthNorm
//
// Key HLSL → MSL translation notes:
//   mul(M, v)                    → M * v
//   Texture2DArray.SampleLevel   → texture2d_array.sample(s, uv, slice, lod:0)
//   RWTexture3D<float>           → texture3d<float, access::read_write>  (R32Float)
//   StructuredBuffer<float3>     → const device float3*
//   uniform float4x4 M[2]        → constant float4x4* M  (2-element array in buffer)
//   SV_DispatchThreadID          → [[thread_position_in_grid]]
//   _m03_m13_m23 (HLSL row0col3…) → .columns[3].xyz  (column-major MSL)

let metalShaderSource = """
#include <metal_stdlib>
using namespace metal;

// ── Uniform structs ────────────────────────────────────────────────────────────

// Matches DepthUniforms in Swift DepthProcessor.swift
struct DepthUniforms {
    float4x4 proj[2];
    float4x4 projInv[2];
    float4x4 view[2];
    float4x4 viewInv[2];
    float2   texSize;
    float2   zParams;    // x=near, y=far
};

// Matches EnvUniforms in Swift DepthProcessor.swift
struct EnvUniforms {
    uint3   voxCount;
    float   voxSize;
    float   voxDist;
    float   voxMin;
    int     numPlayers;
    float   depthDispThresh;
    // playerHeads follow in a separate buffer to keep this struct small
};

struct DilateUniforms {
    int    dilateStepSize;
    float2 texSize;
    float  envVoxDist;
    float  envVoxSize;
};

// ── Coordinate helpers (DepthKit.hlsl) ─────────────────────────────────────────

inline float3 agDepthEyePos(int eye, constant DepthUniforms& u) {
    // HLSL: agDepthViewInv[eye]._m03_m13_m23
    // column-major: translation is in column 3
    return u.viewInv[eye].columns[3].xyz;
}

inline float agDepthNDCToLinear(float depthNDC, int eye, constant DepthUniforms& u) {
    float z = depthNDC * 2.0 - 1.0;
    float A = u.proj[eye].columns[2][2];   // m22
    float B = u.proj[eye].columns[3][2];   // m23
    return abs(B / (z + A));
}

inline float4 agDepthWorldToHCS(float3 worldPos, int eye, constant DepthUniforms& u) {
    return u.proj[eye] * (u.view[eye] * float4(worldPos, 1.0));
}

inline float3 agDepthHCStoNDC(float4 hcs) {
    return (hcs.xyz / hcs.w) * 0.5 + 0.5;
}

inline float3 agDepthWorldToNDC(float3 worldPos, int eye, constant DepthUniforms& u) {
    return agDepthHCStoNDC(agDepthWorldToHCS(worldPos, eye, u));
}

inline float3 agDepthNDCtoWorld(float3 ndc, int eye, constant DepthUniforms& u) {
    float4 hcs    = float4(ndc * 2.0 - 1.0, 1.0);
    float4 worldH = u.viewInv[eye] * (u.projInv[eye] * hcs);
    return worldH.xyz / worldH.w;
}

inline float agDepthSample(float2 uv, int eye,
                            texture2d_array<float, access::sample> depthTex)
{
    constexpr sampler s(coord::normalized, filter::nearest, address::clamp_to_edge);
    return depthTex.sample(s, uv, eye).r;
}

inline float4 agDepthNormalSample(float2 uv, int eye,
                                   texture2d_array<float, access::sample> normTex)
{
    constexpr sampler s(coord::normalized, filter::nearest, address::clamp_to_edge);
    return normTex.sample(s, uv, eye);
}

// ── EnvMapper helpers (EnvMapper.hlsl) ────────────────────────────────────────

#define EMPTY_VOXEL (-1.0f)

inline float3 envVoxelToWorld(uint3 indices, constant EnvUniforms& e) {
    float3 pos = float3(indices);
    pos += 0.5;
    pos -= float3(e.voxCount) / 2.0;
    pos *= e.voxSize;
    return pos;
}

inline float3 envWorldToVoxelFloat(float3 pos, constant EnvUniforms& e) {
    pos /= e.voxSize;
    pos += float3(e.voxCount) / 2.0;
    return pos;
}

inline uint3 envWorldToVoxel(float3 pos, constant EnvUniforms& e) {
    float3 f  = envWorldToVoxelFloat(pos, e);
    uint3  id = uint3(floor(f));
    return clamp(id, uint3(0), e.voxCount);
}

inline float3 envWorldToVoxelUVW(float3 pos, constant EnvUniforms& e) {
    float3 f = envWorldToVoxelFloat(pos, e);
    return saturate(f / float3(e.voxCount));
}

inline float sampleDilatedDepth(float2 uv, texture2d<float, access::sample> dilatedDepth) {
    constexpr sampler s(coord::normalized, filter::nearest, address::clamp_to_edge);
    return dilatedDepth.sample(s, uv).z;
}

// ── Kernel: Clear ──────────────────────────────────────────────────────────────

kernel void kernelClear(
    texture3d<float, access::read_write> volumeRW [[texture(0)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= volumeRW.get_width()  ||
        gid.y >= volumeRW.get_height() ||
        gid.z >= volumeRW.get_depth()) return;
    volumeRW.write(EMPTY_VOXEL, gid);
}

// ── Kernel: DepthNorm (port of DepthNorm.compute) ─────────────────────────────

kernel void kernelDepthNorm(
    texture2d_array<float,  access::sample> depthTex  [[texture(0)]],
    texture2d_array<float, access::write>   normTexRW [[texture(1)]],
    constant DepthUniforms&                 uniforms  [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint w = depthTex.get_width(), h = depthTex.get_height();
    if (gid.x >= w || gid.y >= h) return;

    uint  eye     = gid.z;
    float2 texSz  = float2(w, h);
    float2 uv     = float2(gid.xy) / texSz;

    constexpr sampler s(coord::normalized, filter::nearest, address::clamp_to_edge);
    float d0 = depthTex.sample(s, uv, eye).r;
    float3 w0 = agDepthNDCtoWorld(float3(uv, d0), eye, uniforms);

    uint2 indH = min(gid.xy + uint2(2, 0), uint2(w-1, h-1));
    float2 uvH = float2(indH) / texSz;
    float dH   = depthTex.sample(s, uvH, eye).r;
    float3 wH  = agDepthNDCtoWorld(float3(uvH, dH), eye, uniforms);

    uint2 indV = min(gid.xy + uint2(0, 2), uint2(w-1, h-1));
    float2 uvV = float2(indV) / texSz;
    float dV   = depthTex.sample(s, uvV, eye).r;
    float3 wV  = agDepthNDCtoWorld(float3(uvV, dV), eye, uniforms);

    float3 worldNorm = -normalize(cross(wH - w0, wV - w0));
    normTexRW.write(float4(worldNorm, 1.0), gid.xy, eye);
}

// ── Kernel: InitDepthDilation ─────────────────────────────────────────────────

kernel void kernelInitDepthDilation(
    texture2d_array<float, access::sample> depthTex  [[texture(0)]],
    texture2d<float, access::write>        dilateDst [[texture(1)]],
    uint2 coord [[thread_position_in_grid]])
{
    uint w = depthTex.get_width(), h = depthTex.get_height();
    if (coord.x >= w || coord.y >= h) return;

    constexpr sampler s(coord::normalized, filter::nearest, address::clamp_to_edge);
    float2 uv  = (float2(coord) + 0.5) / float2(w, h);
    float depth = depthTex.sample(s, uv, 0).r;   // eye 0

    float4 val = float4(float2(coord), depth, 0.0);
    dilateDst.write(val, coord);
}

// ── Kernel: DilateDepthStep ───────────────────────────────────────────────────

constant int2 kOffsets[8] = {
    int2( 1,  0), int2( 1,  1), int2( 0,  1), int2(-1,  1),
    int2(-1,  0), int2(-1, -1), int2( 0, -1), int2( 1, -1)
};

kernel void kernelDilateDepthStep(
    texture2d<float, access::read>   dilateSrc [[texture(0)]],
    texture2d<float, access::write>  dilateDst [[texture(1)]],
    constant DilateUniforms&         uniforms  [[buffer(0)]],
    uint2 coord [[thread_position_in_grid]])
{
    uint w = dilateSrc.get_width(), h = dilateSrc.get_height();
    if (coord.x >= w || coord.y >= h) return;

    float4 val          = dilateSrc.read(coord);
    float focalLength   = 0.78;  // approximate; overridden per-frame via EnvUniforms

    for (int i = 0; i < 8; i++) {
        int2 srcCoordI = int2(coord) + kOffsets[i] * uniforms.dilateStepSize;
        if (srcCoordI.x < 0 || srcCoordI.y < 0 ||
            srcCoordI.x >= int(w) || srcCoordI.y >= int(h)) continue;

        uint2 srcCoord = uint2(srcCoordI);
        float4 srcVal  = dilateSrc.read(srcCoord);

        float srcDepthLinear = srcVal.z;   // already NDC depth; reuse as proxy
        float pixelSize  = 2.0 * srcDepthLinear / (focalLength * uniforms.texSize.x);
        float radius     = (uniforms.envVoxDist + uniforms.envVoxSize) / max(pixelSize, 1e-5);

        float2 diff       = srcVal.xy - float2(coord);
        bool withinRadius = dot(diff, diff) < radius * radius;
        bool newMin       = srcVal.z < val.z;
        bool notZero      = srcVal.z != 0.0;

        if (withinRadius && newMin && notZero)
            val = srcVal;
    }

    dilateDst.write(val, coord);
}

// ── Kernel: Integrate (TSDF) ──────────────────────────────────────────────────

#define PLAYER_TOP    0.2f
#define PLAYER_BOTTOM 1.7f
#define PLAYER_RADIUS 0.5f
#define MIN_DOT       0.3f

kernel void kernelIntegrate(
    texture3d<float, access::read_write>    volumeRW    [[texture(0)]],
    texture2d_array<float, access::sample>  depthTex    [[texture(1)]],
    texture2d_array<float, access::sample>  normTex     [[texture(2)]],
    texture2d<float, access::sample>        dilatedDepth[[texture(3)]],
    const device float3*                    frustumVol  [[buffer(0)]],
    constant DepthUniforms&                 depth       [[buffer(1)]],
    constant EnvUniforms&                   env         [[buffer(2)]],
    const device float3*                    playerHeads [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    // View-space position pre-computed on CPU (like froxels)
    float3 vLocalPos = frustumVol[id];
    // Transform to world space using inverse view matrix (eye 0)
    float4 vWorldH   = depth.viewInv[0] * float4(vLocalPos, 1.0);
    float3 vWorldPos = vWorldH.xyz / vWorldH.w;
    uint3  coord     = envWorldToVoxel(vWorldPos, env);
    float3 voxPos    = envVoxelToWorld(coord, env);

    float3 eyePos    = agDepthEyePos(0, depth);
    float3 eyeToVox  = voxPos - eyePos;
    float  voxEyeDist = length(eyeToVox);

    float3 voxNDC    = agDepthWorldToNDC(voxPos, 0, depth);
    float  depthNDC  = agDepthSample(voxNDC.xy, 0, depthTex);
    float4 normSamp  = agDepthNormalSample(voxNDC.xy, 0, normTex);
    float3 depthNorm = normSamp.xyz;
    float3 depthPos  = agDepthNDCtoWorld(float3(voxNDC.xy, depthNDC), 0, depth);

    float3 eyeToDepth  = depthPos - eyePos;
    float  depthEyeDist = length(eyeToDepth);
    float  normDot      = -dot(eyeToVox, depthNorm) / voxEyeDist;

    float sDist = depthEyeDist - voxEyeDist;
    sDist *= saturate(normDot);

    bool  empty       = sDist >= 1.0;
    float sDistNorm   = min(sDist / env.voxDist, 1.0);
    bool  withinBand  = sDistNorm >= -(env.voxMin / env.voxSize);

    float dilatedNDC    = sampleDilatedDepth(voxNDC.xy, dilatedDepth);
    float dilatedLinear = agDepthNDCToLinear(dilatedNDC, 0, depth);
    float depthLinear   = agDepthNDCToLinear(depthNDC,   0, depth);
    float3 voxView      = (depth.view[0] * float4(voxPos, 1.0)).xyz;

    bool acceptable     = depthLinear < dilatedLinear + env.depthDispThresh;
    bool unoccluded     = abs(voxView.z) < dilatedLinear || acceptable;
    bool validNormal    = empty || normDot > MIN_DOT;

    bool outsidePlayers = true;
    for (int p = 0; p < env.numPlayers; p++) {
        float3 ph  = playerHeads[p];
        float2 d2d = vWorldPos.xz - ph.xz;
        if (dot(d2d, d2d) < PLAYER_RADIUS * PLAYER_RADIUS &&
            vWorldPos.y < ph.y + PLAYER_TOP &&
            vWorldPos.y > ph.y - PLAYER_BOTTOM)
        {
            outsidePlayers = false;
        }
    }

    if (withinBand && outsidePlayers && validNormal && unoccluded)
        volumeRW.write(sDistNorm, coord);
}
"""
