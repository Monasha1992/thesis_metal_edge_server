#include <metal_stdlib>
using namespace metal;

struct NormParams {
    float4x4 projInv;
    float4x4 viewInv;
    uint2 texSize;
};

float3 ndcToWorldNorm(float2 uv, float depth, float4x4 projInv, float4x4 viewInv) {
    float4 hcs = float4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    float4 worldH = viewInv * (projInv * hcs);
    return worldH.xyz / worldH.w;
}

kernel void depthNormals(
    texture2d<float, access::read> depthTex [[texture(0)]],
    texture2d<float, access::write> normTex [[texture(1)]],
    constant NormParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.texSize.x || gid.y >= params.texSize.y) return;
    if (gid.x == 0 || gid.y == 0 || gid.x >= params.texSize.x - 1 || gid.y >= params.texSize.y - 1) {
        normTex.write(float4(0, 0, 0, 0), gid);
        return;
    }

    float2 texSize = float2(params.texSize);
    float2 uv = (float2(gid) + 0.5) / texSize;
    float2 pixel = 1.0 / texSize;

    float depth = depthTex.read(gid).r;
    float depthL = depthTex.read(uint2(gid.x - 1, gid.y)).r;
    float depthR = depthTex.read(uint2(gid.x + 1, gid.y)).r;
    float depthD = depthTex.read(uint2(gid.x, gid.y - 1)).r;
    float depthU = depthTex.read(uint2(gid.x, gid.y + 1)).r;

    if (depth == 0.0 || depthL == 0.0 || depthR == 0.0 || depthD == 0.0 || depthU == 0.0) {
        normTex.write(float4(0, 0, 0, 0), gid);
        return;
    }

    float3 posL = ndcToWorldNorm(uv + float2(-pixel.x, 0), depthL, params.projInv, params.viewInv);
    float3 posR = ndcToWorldNorm(uv + float2( pixel.x, 0), depthR, params.projInv, params.viewInv);
    float3 posD = ndcToWorldNorm(uv + float2(0, -pixel.y), depthD, params.projInv, params.viewInv);
    float3 posU = ndcToWorldNorm(uv + float2(0,  pixel.y), depthU, params.projInv, params.viewInv);

    float3 dx = posR - posL;
    float3 dy = posU - posD;
    float3 normal = normalize(cross(dy, dx));

    normTex.write(float4(normal, 1.0), gid);
}
