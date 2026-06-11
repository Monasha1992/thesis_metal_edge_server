#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// VolumeIntegration.metal — Builds and maintains the 3D TSDF volume
//
// WHAT IS A TSDF?
//   TSDF = Truncated Signed Distance Function
//   An invisible 3D grid of voxels (cubes) that covers the room.
//   Each voxel stores ONE number: how far is this voxel from the nearest surface?
//     Positive value → voxel is in front of (outside) a surface
//     Negative value → voxel is behind (inside) a surface
//     Zero           → voxel is exactly on the surface
//   "Truncated" means we only store values within a band around surfaces
//   (e.g. ±0.2m). Voxels far from any surface stay at 0 (unobserved).
//
// WHY USE A TSDF?
//   By accumulating many depth frames into one TSDF volume, we get a stable,
//   smooth 3D model of the room. Individual depth frames are noisy — averaging
//   them in the TSDF smooths out the noise over time.
//
// HOW THE VOLUME IS STORED:
//   A 3D texture with format RG32Float:
//     R channel = TSDF value (updated every frame via a light 0.5 blend)
//     G channel = weight, used only as an "observed / unobserved" flag
//                 (0 = never seen, 1 = seen at least once)
//   Temporal smoothing is a single-constant exponential blend (BLEND = 0.5
//   in the integrate kernel) — see the kernel for the full tuning history.
//
// THIS FILE CONTAINS THREE KERNELS:
//   1. clearVolume   — resets all voxels to unobserved (run once at start)
//   2. integrate     — fuses one depth frame into the volume (run every frame)
//   3. extractNonEmpty — finds all observed voxels (used for debugging)
// ─────────────────────────────────────────────────────────────────────────────

#define EMPTY_VOXEL    -1.0   // Legacy sentinel (no longer used — weight channel replaced it)

// ── Player exclusion cylinder dimensions (metres) ───────────────────────────
// A voxel is excluded if it is inside a capped vertical cylinder around any
// player head. Values match Anaglyph's EnvironmentMapping.compute so that
// Architecture B matches Architecture A behaviour.
#define PLAYER_TOP      0.2   // Cylinder extends this far above the head
#define PLAYER_BOTTOM   1.7   // Cylinder extends this far below the head (torso+legs)
#define PLAYER_RADIUS   0.5   // Cylinder radius in the XZ plane
#define MIN_DOT         0.3   // Minimum surface normal dot product to accept a reading

// Maximum number of player heads sent from the Quest per frame.
// Must match MAX_PLAYERS in EdgeServerClient.cs and DepthFrame.swift.
#define MAX_PLAYERS 8

// Parameters sent from Swift for each frame
struct IntegrateParams {
    float4x4 view;             // Camera view matrix (world → camera space)
    float4x4 proj;             // Camera projection matrix (camera → clip space)
    float4x4 viewInv;          // Inverse view matrix (camera → world space)
    float4x4 projInv;          // Inverse projection matrix (clip → camera space)
    uint3    voxCount;         // Number of voxels in X, Y, Z
    float    voxSize;          // Size of each voxel in metres (e.g. 0.1)
    float    voxDist;          // TSDF truncation distance in metres (e.g. 0.2)
    float    voxMin;           // Minimum distance from camera to integrate (e.g. 0.1m)
    float    depthDispThresh;  // Allowed depth disparity before rejecting a voxel
    int      numPlayers;       // Number of player head positions (for exclusion cylinders)
};

// ── Coordinate conversion helpers ────────────────────────────────────────────

// Extract camera world position from the inverse view matrix (4th column = translation)
float3 eyePos(float4x4 viewInvMat) {
    return float3(viewInvMat[3][0], viewInvMat[3][1], viewInvMat[3][2]);
}

// Transform a world-space position into homogeneous clip space (before perspective divide)
float4 worldToHCS(float3 worldPos, float4x4 viewMat, float4x4 projMat) {
    return projMat * (viewMat * float4(worldPos, 1.0));
}

// Convert homogeneous clip space to NDC (0–1 range UV + depth)
float3 hcsToNDC(float4 hcs) {
    return (hcs.xyz / hcs.w) * 0.5 + 0.5;
}

// World position → NDC (UV + depth, 0–1 range) in one step
float3 worldToNDC(float3 worldPos, float4x4 viewMat, float4x4 projMat) {
    return hcsToNDC(worldToHCS(worldPos, viewMat, projMat));
}

// NDC (UV + depth) → world-space 3D position
float3 ndcToWorld(float3 ndc, float4x4 viewInvMat, float4x4 projInvMat) {
    float4 hcs = float4(ndc * 2.0 - 1.0, 1.0);
    float4 worldH = viewInvMat * (projInvMat * hcs);
    return worldH.xyz / worldH.w;
}

// Convert non-linear NDC depth to linear distance in metres
float ndcToLinear(float depthNDC, float4x4 projMat) {
    float z = depthNDC * 2.0 - 1.0;
    float A = projMat[2][2];
    float B = projMat[2][3];
    return abs(B / (z + A));
}

// Convert a voxel grid index to its world-space centre position
// The volume is centred at world origin (0,0,0)
float3 voxelToWorld(uint3 indices, uint3 voxCount, float voxSize) {
    float3 pos = float3(indices) + 0.5;       // Centre of voxel (not corner)
    pos -= float3(voxCount) / 2.0;            // Shift so volume centre = world origin
    pos *= voxSize;                            // Scale from voxel units to metres
    return pos;
}

// Convert a world-space position to the voxel grid index it falls in
uint3 worldToVoxel(float3 pos, uint3 voxCount, float voxSize) {
    pos /= voxSize;                            // Scale from metres to voxel units
    pos += float3(voxCount) / 2.0;            // Shift so world origin = volume centre
    uint3 id = uint3(floor(pos));
    id = clamp(id, uint3(0), voxCount);       // Clamp to valid range
    return id;
}

// ─────────────────────────────────────────────────────────────────────────────
// clearVolume — Resets every voxel to "unobserved"
//
// Called once when the server starts. Sets weight = 0 for all voxels,
// which marks them as never having been seen. The TSDF value doesn't matter
// until weight > 0.
// ─────────────────────────────────────────────────────────────────────────────
kernel void clearVolume(
    texture3d<float, access::write> volume [[texture(0)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= volume.get_width() || gid.y >= volume.get_height() || gid.z >= volume.get_depth()) return;

    // R=0 (tsdf), G=0 (weight=0 means unobserved)
    volume.write(float4(0.0, 0.0, 0, 0), gid);
}

// ─────────────────────────────────────────────────────────────────────────────
// integrate — Fuses one depth frame into the TSDF volume
//
// This is the core kernel. One GPU thread per frustum voxel position.
//
// HOW IT WORKS:
//   The frustum buffer contains a list of view-space positions that were
//   pre-computed to lie within the camera's field of view. For each position:
//
//   1. Transform from view space to world space
//   2. Find which voxel it maps to in the 3D grid
//   3. Project that voxel onto the depth image to get the measured depth
//   4. Compare: how far is the voxel from the measured surface?
//      signed distance = measured_depth - voxel_depth
//   5. If the voxel is close enough to the surface (within truncation band):
//      blend the new reading with the existing TSDF value using weighted average
//
// TEMPORAL SMOOTHING (light 0.5 exponential blend):
//   newTsdf = 0.5 × old + 0.5 × new. ~One frame of memory: per-frame depth
//   noise halves every frame (no visible mesh "shake" during head motion),
//   while a real scene change reaches ~90 % of its new value in 3-4 frames
//   (~0.4 s). This sits between two rejected extremes that were both tested
//   on-device: heavy weight-capped averaging (10-15 s ghosting on moved
//   objects) and raw direct overwrite (responsive but visibly shimmering).
//   See the integrate kernel's BLEND constant for the full tuning history.
//
// QUALITY GATES (reasons a voxel update is rejected):
//   - Voxel is too far from the surface (outside truncation band)
//   - Surface normal at that pixel is nearly parallel to the view ray
//     (grazing angle = unreliable depth reading)
//   - Voxel is occluded by something closer (dilation occlusion check)
// ─────────────────────────────────────────────────────────────────────────────
kernel void integrate(
    device float3*           frustumVolume [[buffer(0)]],  // Pre-computed frustum voxel positions (view space)
    constant IntegrateParams& params       [[buffer(1)]],  // Camera matrices + volume config
    // Player head world positions — 3 floats per player × MAX_PLAYERS.
    // Only the first params.numPlayers entries are valid; the rest are zero.
    constant float*          playerHeads   [[buffer(2)]],
    texture2d<float, access::read>  depthTex     [[texture(0)]],  // Original depth image
    texture2d<float, access::read>  normTex      [[texture(1)]],  // Surface normals
    texture2d<float, access::read>  dilatedDepth [[texture(2)]],  // Dilated depth (for occlusion check)
    texture3d<float, access::read_write> volume  [[texture(3)]],  // The TSDF volume (read + write)
    uint id [[thread_position_in_grid]]
) {
    // Each thread handles one frustum voxel position
    float3 vLocalPos = frustumVolume[id];

    // Transform this view-space position to world space using the camera's pose
    float3 vWorldPos = (params.viewInv * float4(vLocalPos, 1.0)).xyz;

    // Find which voxel in the grid this world position falls into
    uint3 coord = worldToVoxel(vWorldPos, params.voxCount, params.voxSize);

    // Get the world-space centre of that voxel
    float3 voxPos = voxelToWorld(coord, params.voxCount, params.voxSize);

    // Vector from camera to this voxel, and its distance
    float3 eye = eyePos(params.viewInv);
    float3 eyeToVox = voxPos - eye;
    float voxEyeDist = length(eyeToVox);

    // Project the voxel onto the depth image to find the corresponding pixel
    float3 voxNDC = worldToNDC(voxPos, params.view, params.proj);
    uint2 depthCoord = uint2(voxNDC.xy * float2(depthTex.get_width(), depthTex.get_height()));
    depthCoord = clamp(depthCoord, uint2(0), uint2(depthTex.get_width()-1, depthTex.get_height()-1));

    // Sample the measured depth and surface normal at that pixel
    float depthNDC   = depthTex.read(depthCoord).r;
    float3 depthNorm = normTex.read(depthCoord).xyz;

    // Convert the measured depth pixel to a 3D world position
    float3 depthPos = ndcToWorld(float3(voxNDC.xy, depthNDC), params.viewInv, params.projInv);

    // Distance from camera to the measured surface at this pixel
    float3 eyeToDepth = depthPos - eye;
    float depthEyeDist = length(eyeToDepth);

    // How aligned is the surface normal with our view direction?
    // normDot ≈ 1.0 = surface faces camera (reliable)
    // normDot ≈ 0.0 = surface is edge-on (unreliable)
    float normDot = -dot(eyeToVox, depthNorm) / voxEyeDist;

    // Signed distance: positive = voxel is in front of surface, negative = behind
    // Scaled by normDot to weight confidence by surface angle
    float sDist = depthEyeDist - voxEyeDist;
    sDist *= saturate(normDot);

    // "empty" means the voxel is so far in front of the surface it's clearly free space
    bool empty = sDist >= 1.0;

    // Normalise signed distance to [-1, 1] range for storage
    float sDistNorm = min(sDist / params.voxDist, 1.0);

    // Only update voxels within the truncation band around the surface
    bool withinBand = sDistNorm >= -params.voxMin / params.voxSize;

    // ── Dilation occlusion check ──────────────────────────────────────────────
    // Uses the dilated depth to reject voxels that are occluded by geometry
    // that the sensor might have missed (holes filled by dilation)
    float dilatedDepthNDC    = dilatedDepth.read(depthCoord).z;
    float dilatedDepthLinear = ndcToLinear(dilatedDepthNDC, params.proj);
    float depthLinear        = ndcToLinear(depthNDC, params.proj);
    float3 voxView           = (params.view * float4(voxPos, 1.0)).xyz;
    bool acceptableDepthDisparity = depthLinear < dilatedDepthLinear + params.depthDispThresh;
    bool unoccludedByDilation     = abs(voxView.z) < dilatedDepthLinear || acceptableDepthDisparity;

    // Reject voxels where the surface normal is too edge-on (unreliable reading)
    // Exception: if this voxel is clearly empty space (far in front), always accept
    //
    // Threshold tuning history:
    //   0.3 (~73° cone) — visibly shrank curved/round objects. The curved
    //                     sides of cylinders, pillars, table legs are at
    //                     grazing angles from any single viewpoint, so most
    //                     of their surface voxels were rejected and never
    //                     made it into the mesh.
    //   0.1 (~84° cone) — current. Keeps almost every observation. The
    //                     `sDist *= saturate(normDot)` line above already
    //                     scales grazing-angle observations down to their
    //                     true perpendicular distance, so accepting them
    //                     doesn't blur surfaces — it just stops discarding
    //                     valid geometry. Drop to 0.0 to accept everything
    //                     including pure edge-on rays (noisy, not recommended).
    bool validSurfaceNormal = empty || normDot > 0.1;

    // ── Player exclusion cylinders ────────────────────────────────────────────
    // Skip voxels that fall inside a capped vertical cylinder around any player
    // head. This prevents the player's own body (and any visible teammates) from
    // being fused into the reconstructed room mesh.
    //
    // For each of the first params.numPlayers slots:
    //   - Read 3 floats (x, y, z) from playerHeads
    //   - Check if vWorldPos is within PLAYER_RADIUS in XZ and the vertical band
    //     [head.y - PLAYER_BOTTOM, head.y + PLAYER_TOP]
    bool outsidePlayers = true;
    int playerCount = min(params.numPlayers, (int)MAX_PLAYERS);
    for (int p = 0; p < playerCount; p++) {
        float3 playerPos = float3(
            playerHeads[p * 3 + 0],
            playerHeads[p * 3 + 1],
            playerHeads[p * 3 + 2]
        );

        float2 diff = vWorldPos.xz - playerPos.xz;
        bool insideRadius   = dot(diff, diff) < (PLAYER_RADIUS * PLAYER_RADIUS);
        bool insideVertical = (vWorldPos.y < playerPos.y + PLAYER_TOP) &&
                              (vWorldPos.y > playerPos.y - PLAYER_BOTTOM);

        if (insideRadius && insideVertical) {
            outsidePlayers = false;
            break;
        }
    }

    // ── Update voxel if all quality gates pass ────────────────────────────────
    if (withinBand && unoccludedByDilation && validSurfaceNormal && outsidePlayers) {
        // ── Light exponential blend (anti-shake) ──────────────────────────────
        // newTsdf = BLEND × old + (1 − BLEND) × new.
        // At 0.5 this has ~one frame of memory: per-frame depth noise halves
        // every frame (the visible mesh stops "shaking" while the head moves),
        // while a real scene change still reaches ~90 % of its new value in
        // 3-4 frames (~0.4 s at 8 Hz) — nothing like the multi-second
        // ghosting of the old weight-capped running average.
        //
        // History of this kernel (each stage tested on-device):
        //   weighted avg, cap = 30  → very smooth, 10-15 s ghost on moved objects
        //   cap 8 + tight carving   → smooth, still noticeably laggy
        //   cap 3 + broad carving   → close but still felt slow
        //   direct overwrite        → maximally responsive, but per-frame noise
        //                             made the whole mesh visibly shake/shimmer,
        //                             worst while the head was moving
        //   0.5 blend (current)     → kills the shimmer, keeps ~0.4 s response
        //
        // Tuning: raise BLEND toward 0.7 for more smoothing (slower response),
        // lower toward 0.0 for the raw direct-overwrite behaviour.
        //
        // Weight (G channel) stays a pure "has this voxel ever been observed"
        // flag: 0 = never, 1 = yes. First observation skips the blend so a
        // fresh voxel snaps straight to its measured value instead of blending
        // with the unobserved-sentinel 0.
        const float BLEND = 0.5;

        float2 old = volume.read(uint3(coord)).rg;
        float newTsdf = (old.g < 0.5)
                        ? sDistNorm
                        : mix(sDistNorm, old.r, BLEND);

        volume.write(float4(newTsdf, 1.0, 0, 0), coord);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// extractNonEmpty — Collects all observed voxels into a flat list
//
// Used for debugging and bounding box computation.
// A voxel is "observed" if its weight > 0.5 (has been seen at least once).
// Outputs up to maxOutput voxels with their coordinates and TSDF values.
// ─────────────────────────────────────────────────────────────────────────────
struct VoxelData {
    uint  coordX;
    uint  coordY;
    uint  coordZ;
    float value;   // TSDF value at this voxel
};

kernel void extractNonEmpty(
    texture3d<float, access::read> volume [[texture(0)]],
    device atomic_uint* counter           [[buffer(0)]],  // Atomic counter for thread-safe indexing
    device VoxelData*   output            [[buffer(1)]],  // Output array of voxel data
    constant uint&      maxOutput         [[buffer(2)]],  // Maximum entries to write (buffer limit)
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= volume.get_width() || gid.y >= volume.get_height() || gid.z >= volume.get_depth()) return;

    float2 rg = volume.read(gid).rg;

    // Skip voxels that have never been observed (weight < 0.5)
    if (rg.g < 0.5) return;

    float val = rg.r;

    // Atomically grab the next output slot (thread-safe increment)
    uint idx = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
    if (idx >= maxOutput) return;  // Buffer full — drop this voxel

    output[idx].coordX = gid.x;
    output[idx].coordY = gid.y;
    output[idx].coordZ = gid.z;
    output[idx].value  = val;
}
