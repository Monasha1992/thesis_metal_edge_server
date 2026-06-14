#include <metal_stdlib>
using namespace metal;

// ─────────────────────────────────────────────────────────────────────────────
// SurfaceNets.metal — Extracts a triangle mesh from the TSDF volume
//
// WHAT THIS DOES:
//   Takes the 3D TSDF volume (built by VolumeIntegration.metal) and turns it
//   into a triangle mesh that Unity can render.
//
// WHAT IS SURFACE NETS?
//   Surface Nets is a meshing algorithm. It finds where the TSDF values in
//   the volume change sign (positive → negative or negative → positive).
//   That sign change means a physical surface exists between those two voxels.
//   The algorithm places one vertex per voxel cell that contains a surface,
//   then connects neighbouring vertices into triangles.
//
//   Compared to Marching Cubes (the more common algorithm), Surface Nets:
//   - Places vertices more centrally (smoother mesh)
//   - Generates fewer triangles (one quad per crossing instead of many)
//   - Does not use a lookup table
//
// TWO-PASS APPROACH:
//   Pass 1 (surfaceNetsVertices): For each voxel cell, check if it contains
//     a surface crossing. If so, compute a vertex position and store it.
//     Also write to coordVertMap: a lookup table of (cell → vertex index).
//
// COORDVERTMAP IS REGION-RELATIVE:
//   The map is indexed by LOCAL cell coordinates within [regionMin, regionMax),
//   sized regionSize.x*y*z — NOT by absolute volume coordinates. This keeps the
//   buffer small (a 34³ chunk region needs ~157 KB instead of a full-volume
//   1024×256×1024 map needing ~1 GB) so per-chunk meshing can clear and reuse
//   it cheaply. Both passes use the same region so the indexing is consistent.
//   A neighbour lookup that would fall outside the region is treated as
//   INVALID_VERTEX — the adjacent chunk emits those seam quads instead (chunk
//   regions overlap by one cell, so every seam quad is emitted by exactly the
//   dispatches whose region fully contains its four cells).
//
//   Pass 2 (surfaceNetsIndices): For each voxel cell that has a vertex,
//     look at its 3 axis-aligned neighbours. If there's a surface crossing
//     between them, form a quad (2 triangles) using the 4 surrounding vertices.
//
// HOW VERTICES ARE POSITIONED:
//   For each of the 12 edges of a voxel cube, check if the two endpoints
//   have opposite signs (one inside surface, one outside). If yes, that edge
//   "crosses" the surface. Interpolate along the edge to find where the sign
//   flips to zero — that's the crossing point.
//   The vertex is placed at the average of all crossing points for that cell.
// ─────────────────────────────────────────────────────────────────────────────

#define EMPTY_VOXEL    -1.0   // Legacy (not used — weight channel handles empty detection)
#define INVALID_VERTEX -1     // Sentinel: this voxel has no vertex

// One vertex: world-space position + surface normal
// Using packed_float3 (24 bytes) NOT float3 (which is padded to 32 bytes in Metal)
struct Vertex {
    packed_float3 pos;   // World-space position (X, Y, Z)
    packed_float3 norm;  // Surface normal direction (X, Y, Z)
};

// Parameters for both mesh kernels
struct MeshParams {
    int3  voxCount;   // Total volume dimensions (1024×256×1024 in this project)
    float voxSize;    // Size of each voxel in metres (e.g. 0.1)
    int3  regionMin;  // Start of the region to mesh (voxel coordinates)
    int3  regionMax;  // End of the region to mesh (voxel coordinates)
};

// The 8 corners of a unit voxel cube (local offsets from the cell origin)
constant int3 cornerOffsets[8] = {
    int3(0, 0, 0), int3(1, 0, 0), int3(1, 0, 1), int3(0, 0, 1),
    int3(0, 1, 0), int3(1, 1, 0), int3(1, 1, 1), int3(0, 1, 1)
};

// The 12 edges of a unit cube, defined by pairs of corner indices
// edgeA[i] and edgeB[i] are the two endpoints of edge i
constant int edgeA[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 };
constant int edgeB[12] = { 1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7 };

// Convert a 3D voxel coordinate to a flat 1D index (for the coordVertMap lookup table)
int flattenCoord(int3 coord, int3 voxCount) {
    return coord.x + coord.y * voxCount.x + coord.z * voxCount.x * voxCount.y;
}

// Convert a flat 1D index back to a 3D voxel coordinate
int3 unflattenCoord(int idx, int3 size) {
    int z = idx / (size.x * size.y);
    int y = (idx / size.x) % size.y;
    int x = idx % size.x;
    return int3(x, y, z);
}

// Read a voxel's TSDF value from the volume
// Returns 0.0 for out-of-bounds or unobserved voxels (weight < 0.5)
// Returning 0.0 (not -1.0) prevents false surface crossings at volume boundaries
float readVolume(texture3d<float, access::read> volume, int3 coord, int3 voxCount) {
    if (any(coord < 0) || any(coord >= voxCount)) return 0.0;
    float2 rg = volume.read(uint3(coord)).rg;
    if (rg.g < 0.5) return 0.0;  // Unobserved voxel — treat as empty
    return rg.r;
}

// Convert a voxel-space coordinate to world-space position
// The volume is centred at world origin (0,0,0)
float3 coordToPos(float3 coord, int3 voxCount, float voxSize) {
    return (coord - float3(voxCount) * 0.5) * voxSize;
}

// ─────────────────────────────────────────────────────────────────────────────
// surfaceNetsVertices — Pass 1: Generate one vertex per surface voxel cell
//
// For each voxel cell in the meshing region:
//   1. Check all 12 edges of the cell for sign changes (surface crossings)
//   2. If at least 3 crossings found: compute vertex position + normal
//   3. Store vertex in the vertex buffer, record its index in coordVertMap
//   4. If fewer than 3 crossings: mark as INVALID_VERTEX (no surface here)
// ─────────────────────────────────────────────────────────────────────────────
kernel void surfaceNetsVertices(
    texture3d<float, access::read> volume    [[texture(0)]],  // TSDF volume
    device Vertex*          vertices         [[buffer(0)]],   // Output: vertex positions + normals
    device atomic_int*      vertexCounter    [[buffer(1)]],   // Atomic counter for vertex index allocation
    device int*             coordVertMap     [[buffer(2)]],   // Map: voxel coord → vertex index
    constant MeshParams&    params           [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    int3 voxCount  = params.voxCount;
    int3 regionSize  = params.regionMax - params.regionMin;
    int  regionTotal = regionSize.x * regionSize.y * regionSize.z;
    if (int(tid) >= regionTotal) return;

    // Convert flat thread ID to 3D voxel coordinate within the region
    int3 localCoord = unflattenCoord(int(tid), regionSize);
    int3 coord = localCoord + params.regionMin;

    // Skip cells at the last voxel in each dimension — they need a +1 neighbour
    if (coord.x >= voxCount.x - 1 ||
        coord.y >= voxCount.y - 1 ||
        coord.z >= voxCount.z - 1) {
        coordVertMap[flattenCoord(localCoord, regionSize)] = INVALID_VERTEX;
        return;
    }

    float3 posAccum  = float3(0.0);  // Accumulates crossing positions (averaged for vertex)
    float3 dirAccum  = float3(0.0);  // Accumulates gradient direction (used for normal)
    int    numCrossings = 0;

    // Check all 12 edges of this voxel cell
    for (int e = 0; e < 12; e++) {
        int3 coordA = coord + cornerOffsets[edgeA[e]];
        int3 coordB = coord + cornerOffsets[edgeB[e]];

        float valA = readVolume(volume, coordA, voxCount);
        float valB = readVolume(volume, coordB, voxCount);

        float change = valA - valB;

        // Accumulate gradient direction — used to estimate the surface normal
        dirAccum += float3(coordA - coordB) * change;

        // A crossing occurs when one endpoint is positive and the other negative
        // Strict check: 0.0 (unobserved) does NOT count as either side
        bool doesCross = (valA > 0 && valB < 0) || (valA < 0 && valB > 0);
        if (doesCross) {
            // Interpolate along the edge to find where the value is exactly 0
            // t=0 → at coordA, t=1 → at coordB
            float t = valA / change;
            float3 crossingCoord = float3(coordA) + t * float3(coordB - coordA);
            posAccum += crossingCoord;
            numCrossings++;
        }
    }

    // Require at least 3 crossings — fewer means this is a noisy or corner voxel
    // that wouldn't produce a good-looking surface
    if (numCrossings < 3) {
        coordVertMap[flattenCoord(localCoord, regionSize)] = INVALID_VERTEX;
        return;
    }

    // Vertex position = average of all crossing points, converted to world space
    posAccum /= float(numCrossings);
    float3 pos = coordToPos(posAccum, params.voxCount, params.voxSize);

    // Normal = normalised gradient direction
    // If dirAccum is near zero (degenerate case), default to pointing up
    float3 norm = length(dirAccum) > 1e-6 ? normalize(dirAccum) : float3(0, 1, 0);

    // Atomically grab the next available slot in the vertex buffer
    int vertIdx = atomic_fetch_add_explicit(vertexCounter, 1, memory_order_relaxed);
    vertices[vertIdx] = Vertex { pos, norm };

    // Record this cell's vertex index (region-relative) so Pass 2 can look it up
    coordVertMap[flattenCoord(localCoord, regionSize)] = vertIdx;
}

// ─────────────────────────────────────────────────────────────────────────────
// surfaceNetsIndices — Pass 2: Connect vertices into triangles
//
// For each voxel cell that has a valid vertex (from Pass 1):
//   Check 3 axis-aligned edges (X, Y, Z directions).
//   For each edge where the TSDF changes sign (surface crossing):
//     Form a quad using the 4 vertex-carrying voxels that surround that edge.
//     Split the quad into 2 triangles.
//     Winding order (which side faces out) is determined by the sign of valA.
// ─────────────────────────────────────────────────────────────────────────────
kernel void surfaceNetsIndices(
    texture3d<float, access::read> volume [[texture(0)]],  // TSDF volume (for crossing checks)
    device int*          coordVertMap     [[buffer(0)]],   // Map: voxel coord → vertex index (from Pass 1)
    device uint*         triangles        [[buffer(1)]],   // Output: triangle index list
    device atomic_int*   triCounter       [[buffer(2)]],   // Atomic counter for triangle index allocation
    constant MeshParams& params           [[buffer(3)]],
    device Vertex*       vertices         [[buffer(4)]],   // Vertex buffer (not used here, kept for future)
    uint tid [[thread_position_in_grid]]
) {
    int3 voxCount    = params.voxCount;
    int3 regionSize  = params.regionMax - params.regionMin;
    int  regionTotal = regionSize.x * regionSize.y * regionSize.z;
    if (int(tid) >= regionTotal) return;

    int3 localCoord = unflattenCoord(int(tid), regionSize);
    int3 coord = localCoord + params.regionMin;

    // This voxel must have a vertex (from Pass 1) to emit any triangles.
    // Map lookups are region-relative (see header comment).
    int vertA = coordVertMap[flattenCoord(localCoord, regionSize)];
    if (vertA == INVALID_VERTEX) return;

    // Check all 3 axis-aligned directions for surface crossings
    // For each crossing, form a quad with the 4 voxels surrounding that edge
    int3 axes[3] = { int3(1,0,0), int3(0,1,0), int3(0,0,1) };
    int3 d1s[3]  = { int3(0,0,1), int3(1,0,0), int3(0,1,0) };  // Two perpendicular directions
    int3 d2s[3]  = { int3(0,1,0), int3(0,0,1), int3(1,0,0) };  // for finding the 4 quad corners

    for (int ax = 0; ax < 3; ax++) {
        int3 axis = axes[ax];
        int3 d1   = d1s[ax];
        int3 d2   = d2s[ax];

        // We need neighbours at -d1 and -d2 — skip if that would step outside
        // the meshing REGION (covers the volume edge too, since the map is
        // region-relative and the neighbouring chunk emits the seam quads
        // its own region fully contains).
        if (any(localCoord - d1 < 0) || any(localCoord - d2 < 0)) continue;

        // Check if there's a surface crossing along this axis
        float valA = readVolume(volume, coord,        voxCount);
        float valB = readVolume(volume, coord + axis, voxCount);

        // No crossing — same sign on both sides
        if ((valA < 0) == (valB < 0)) continue;

        // The 4 voxel cells that share this edge — each contributes one vertex
        //   a = this cell
        //   b = one step back in d1 direction
        //   c = one step back in both d1 and d2
        //   d = one step back in d2 direction
        int a = vertA;
        int b = coordVertMap[flattenCoord(localCoord - d1,        regionSize)];
        int c = coordVertMap[flattenCoord(localCoord - (d1 + d2), regionSize)];
        int d = coordVertMap[flattenCoord(localCoord - d2,        regionSize)];

        // All 4 corners must have vertices — skip if any are missing
        if (b == INVALID_VERTEX || c == INVALID_VERTEX || d == INVALID_VERTEX) continue;

        // Atomically reserve 6 index slots (2 triangles × 3 indices each)
        int triIdx = atomic_fetch_add_explicit(triCounter, 6, memory_order_relaxed);

        // Winding order: flip based on which side is "inside" (negative TSDF)
        // This ensures the triangle normal always faces outward from the surface
        if (valA < 0) {
            // valA is inside — wind counter-clockwise to face outward
            triangles[triIdx + 0] = uint(c);
            triangles[triIdx + 1] = uint(b);
            triangles[triIdx + 2] = uint(a);
            triangles[triIdx + 3] = uint(d);
            triangles[triIdx + 4] = uint(c);
            triangles[triIdx + 5] = uint(a);
        } else {
            // valA is outside — wind clockwise
            triangles[triIdx + 0] = uint(a);
            triangles[triIdx + 1] = uint(c);
            triangles[triIdx + 2] = uint(d);
            triangles[triIdx + 3] = uint(a);
            triangles[triIdx + 4] = uint(b);
            triangles[triIdx + 5] = uint(c);
        }
    }
}
