#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// DepthDilation.metal — Fills holes in the depth image
//
// WHAT THIS DOES:
//   The Quest 3 depth sensor has gaps — pixels with no valid depth reading.
//   This happens on reflective surfaces, thin objects, and at depth discontinuities.
//   If we leave gaps in the depth image, they create holes in the final mesh.
//
//   This shader fills those gaps by "dilating" valid depth values outward into
//   invalid (zero) regions — like spreading ink on wet paper.
//
// HOW IT WORKS (Jump-Flood Dilation):
//   Instead of checking every neighbour pixel (slow), we use a jump-flood
//   approach: run 8 passes with step sizes 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
//   Each pass lets valid depth values "jump" over large gaps in one step.
//   This fills gaps up to 256 pixels wide in just 8 passes instead of 256.
//
//   Each pixel stores: (original_x, original_y, depth_value, unused)
//   so we always know where the depth value originally came from.
//
// ACCEPTANCE RULE:
//   A neighbour's depth is only accepted if:
//   - It is within a physically meaningful radius (based on actual depth distance)
//   - It is closer than the current value (nearer depth wins)
//   - It is not zero (zero = no reading)
// ─────────────────────────────────────────────────────────────────────────────

struct DilationParams {
    float4x4 proj;      // Camera projection matrix (used to convert NDC depth to metres)
    uint2 texSize;      // Width and height of the depth image in pixels
    float voxDist;      // TSDF truncation distance in metres (e.g. 0.2m)
    float voxSize;      // Size of one voxel in metres (e.g. 0.1m)
    int stepSize;       // Current jump distance in pixels (256, 128, 64... 1)
};

// 8 neighbour directions to sample (N, NE, E, SE, S, SW, W, NW)
constant int2 kOffsets[8] = {
    int2(1, 0), int2(1, 1), int2(0, 1), int2(-1, 1),
    int2(-1, 0), int2(-1, -1), int2(0, -1), int2(1, -1)
};

// ─────────────────────────────────────────────────────────────────────────────
// initDepthDilation — First pass: copy depth into the ping-pong buffer
//
// Converts the input depth texture into the 4-channel format used by dilation:
//   R = original pixel X position
//   G = original pixel Y position
//   B = depth value (NDC)
//   A = unused
//
// This lets each pixel "remember" where its depth value came from, even after
// being spread to neighbouring pixels in later passes.
// ─────────────────────────────────────────────────────────────────────────────
kernel void initDepthDilation(
    texture2d<float, access::read>  depthTex [[texture(0)]],
    texture2d<float, access::write> dest     [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= depthTex.get_width() || gid.y >= depthTex.get_height()) return;

    float depth = depthTex.read(gid).r;

    // Store: (pixel_x, pixel_y, depth, 0)
    float4 val = float4(float2(gid), depth, 0.0);
    dest.write(val, gid);
}

// ─────────────────────────────────────────────────────────────────────────────
// dilateDepthStep — One dilation pass at a given step size
//
// For each pixel, look at 8 neighbours at distance `stepSize` pixels away.
// If a neighbour has a valid (non-zero) depth that is:
//   - closer than what this pixel currently holds
//   - within a physically reasonable fill radius
// ...then adopt that depth value.
//
// Called 8 times with step sizes: 256, 128, 64, 32, 16, 8, 4, 2 (then 1 implied)
// Uses ping-pong buffers (src → dest, then swap) so reads and writes don't conflict.
// ─────────────────────────────────────────────────────────────────────────────
kernel void dilateDepthStep(
    texture2d<float, access::read>  src  [[texture(0)]],
    texture2d<float, access::write> dest [[texture(1)]],
    constant DilationParams& params      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.texSize.x || gid.y >= params.texSize.y) return;

    float4 val = src.read(gid);  // Current best value for this pixel

    // Focal length from the projection matrix (tells us pixels-per-metre at distance 1)
    float focalLength = params.proj[0][0];

    for (int i = 0; i < 8; i++) {
        // Sample a neighbour pixel at stepSize distance in direction i
        uint2 srcCoord = uint2(int2(gid) + kOffsets[i] * params.stepSize);

        bool withinBounds = srcCoord.x < params.texSize.x && srcCoord.y < params.texSize.y;
        if (!withinBounds) continue;

        float4 srcVal = src.read(srcCoord);

        // Convert the neighbour's NDC depth to linear metres
        // NDC depth is non-linear (perspective projection), we need real distance
        float A = params.proj[2][2];
        float B = params.proj[2][3];
        float srcZ = srcVal.z * 2.0 - 1.0;
        float srcDepthLinear = abs(B / (srcZ + A));  // Distance in metres

        // At this depth, how large is one pixel in metres?
        // Used to determine how far (in pixels) we should allow filling
        float pixelSize = 2.0 * srcDepthLinear / (focalLength * float(params.texSize.x));

        // Maximum fill radius in pixels — based on the TSDF band size
        // Farther surfaces have larger pixels so the radius is bigger
        float radius = (params.voxDist + params.voxSize) / pixelSize;

        // Distance (in pixels) from the neighbour's original source position to here
        float2 diff = srcVal.xy - float2(gid);
        bool withinRadius = dot(diff, diff) < radius * radius;

        // Only accept if: within radius AND closer depth AND not a gap (zero)
        bool newMinimum = srcVal.z < val.z;
        bool notZero = srcVal.z != 0.0;

        if (withinRadius && newMinimum && notZero) {
            val = srcVal;  // Adopt this neighbour's depth value
        }
    }

    dest.write(val, gid);
}
