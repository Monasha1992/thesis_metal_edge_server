#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// DepthNormal.metal — Estimates surface normals from the depth image
//
// WHAT THIS DOES:
//   A depth image tells us HOW FAR each point is from the camera, but not
//   WHICH WAY the surface at that point is facing. This shader computes that
//   facing direction — called a "surface normal" — for every pixel.
//
// WHY NORMALS ARE NEEDED:
//   The TSDF integration step uses normals to:
//   1. Judge how reliable a depth reading is (a surface seen head-on is more
//      reliable than one seen at a grazing angle)
//   2. Gate which voxels get updated (surfaces seen edge-on are ignored)
//
// HOW IT WORKS:
//   For each pixel, look at its 4 neighbours (left, right, up, down).
//   Convert all 5 depth values to 3D world-space positions.
//   Compute two tangent vectors across the surface:
//     dx = right_position - left_position   (horizontal tangent)
//     dy = up_position - down_position      (vertical tangent)
//   The normal = cross product of dy and dx
//   (note: dy × dx, not dx × dy — the order matters for the correct direction
//    due to Unity's left-handed coordinate system)
//
// OUTPUT:
//   An RGBA texture where RGB = the normal direction (XYZ), A = 1.0
//   Pixels with any missing depth neighbours get normal (0,0,0) — invalid.
// ─────────────────────────────────────────────────────────────────────────────

struct NormParams {
    float4x4 projInv;   // Inverse projection matrix — converts screen coords to view space
    float4x4 viewInv;   // Inverse view matrix — converts view space to world space
    uint2 texSize;      // Width and height of the depth image
};

// Converts a 2D screen position + depth value to a 3D world-space position
float3 ndcToWorldNorm(float2 uv, float depth, float4x4 projInv, float4x4 viewInv) {
    // Convert UV + depth to clip space (NDC)
    float4 hcs = float4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    // Unproject through view and projection inverses to get world position
    float4 worldH = viewInv * (projInv * hcs);
    return worldH.xyz / worldH.w;
}

// ─────────────────────────────────────────────────────────────────────────────
// depthNormals — Main compute kernel, one thread per pixel
// ─────────────────────────────────────────────────────────────────────────────
kernel void depthNormals(
    texture2d<float, access::read>  depthTex [[texture(0)]],  // Input: original (undilated) depth image
    texture2d<float, access::write> normTex  [[texture(1)]],  // Output: normal map
    constant NormParams& params              [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.texSize.x || gid.y >= params.texSize.y) return;

    // Border pixels can't have all 4 neighbours — write invalid normal
    if (gid.x == 0 || gid.y == 0 || gid.x >= params.texSize.x - 1 || gid.y >= params.texSize.y - 1) {
        normTex.write(float4(0, 0, 0, 0), gid);
        return;
    }

    float2 texSize = float2(params.texSize);
    float2 uv = (float2(gid) + 0.5) / texSize;   // UV of this pixel (0-1 range)
    float2 pixel = 1.0 / texSize;                 // Size of one pixel in UV space

    // Read depth at this pixel and its 4 neighbours
    float depth  = depthTex.read(gid).r;
    float depthL = depthTex.read(uint2(gid.x - 1, gid.y)).r;
    float depthR = depthTex.read(uint2(gid.x + 1, gid.y)).r;
    float depthD = depthTex.read(uint2(gid.x, gid.y - 1)).r;
    float depthU = depthTex.read(uint2(gid.x, gid.y + 1)).r;

    // If any neighbour has no depth reading, we can't compute a reliable normal
    if (depth == 0.0 || depthL == 0.0 || depthR == 0.0 || depthD == 0.0 || depthU == 0.0) {
        normTex.write(float4(0, 0, 0, 0), gid);
        return;
    }

    // Convert each depth pixel to a 3D world-space point
    float3 posL = ndcToWorldNorm(uv + float2(-pixel.x, 0), depthL, params.projInv, params.viewInv);
    float3 posR = ndcToWorldNorm(uv + float2( pixel.x, 0), depthR, params.projInv, params.viewInv);
    float3 posD = ndcToWorldNorm(uv + float2(0, -pixel.y), depthD, params.projInv, params.viewInv);
    float3 posU = ndcToWorldNorm(uv + float2(0,  pixel.y), depthU, params.projInv, params.viewInv);

    // Two tangent vectors along the surface
    float3 dx = posR - posL;  // Horizontal tangent (left → right)
    float3 dy = posU - posD;  // Vertical tangent (down → up)

    // Cross product gives the perpendicular direction = surface normal
    // dy × dx (not dx × dy) because Unity uses a left-handed coordinate system
    float3 normal = normalize(cross(dy, dx));

    normTex.write(float4(normal, 1.0), gid);
}
