#include <metal_stdlib>
using namespace metal;

#define EMPTY_VOXEL -1.0
#define INVALID_VERTEX -1

struct Vertex {
    packed_float3 pos;
    packed_float3 norm;
};

struct MeshParams {
    int3 voxCount;
    float voxSize;
    int3 regionMin;
    int3 regionMax;
};

constant int3 cornerOffsets[8] = {
    int3(0, 0, 0),
    int3(1, 0, 0),
    int3(1, 0, 1),
    int3(0, 0, 1),
    int3(0, 1, 0),
    int3(1, 1, 0),
    int3(1, 1, 1),
    int3(0, 1, 1)
};

constant int edgeA[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 };
constant int edgeB[12] = { 1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7 };

int flattenCoord(int3 coord, int3 voxCount) {
    return coord.x + coord.y * voxCount.x + coord.z * voxCount.x * voxCount.y;
}

int3 unflattenCoord(int idx, int3 size) {
    int z = idx / (size.x * size.y);
    int y = (idx / size.x) % size.y;
    int x = idx % size.x;
    return int3(x, y, z);
}

float readVolume(texture3d<float, access::read> volume, int3 coord, int3 voxCount) {
    if (any(coord < 0) || any(coord >= voxCount)) return 0.0;
    float2 rg = volume.read(uint3(coord)).rg;
    if (rg.g < 0.5) return 0.0; // unobserved (weight == 0)
    return rg.r;
}

float3 coordToPos(float3 coord, int3 voxCount, float voxSize) {
    return (coord - float3(voxCount) * 0.5) * voxSize;
}

kernel void surfaceNetsVertices(
    texture3d<float, access::read> volume [[texture(0)]],
    device Vertex* vertices [[buffer(0)]],
    device atomic_int* vertexCounter [[buffer(1)]],
    device int* coordVertMap [[buffer(2)]],
    constant MeshParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    int3 voxCount = params.voxCount;
    int3 regionSize = params.regionMax - params.regionMin;
    int regionTotal = regionSize.x * regionSize.y * regionSize.z;
    if (int(tid) >= regionTotal) return;

    int3 localCoord = unflattenCoord(int(tid), regionSize);
    int3 coord = localCoord + params.regionMin;

    // Skip last edge in each dimension
    if (coord.x >= voxCount.x - 1 ||
        coord.y >= voxCount.y - 1 ||
        coord.z >= voxCount.z - 1) {
        coordVertMap[flattenCoord(coord, voxCount)] = INVALID_VERTEX;
        return;
    }

    float3 posAccum = float3(0.0);
    float3 dirAccum = float3(0.0);
    int numCrossings = 0;

    for (int e = 0; e < 12; e++) {
        int3 coordA = coord + cornerOffsets[edgeA[e]];
        int3 coordB = coord + cornerOffsets[edgeB[e]];

        float valA = readVolume(volume, coordA, voxCount);
        float valB = readVolume(volume, coordB, voxCount);

        float change = valA - valB;
        dirAccum += float3(coordA - coordB) * change;

        bool doesCross = (valA > 0 && valB < 0) || (valA < 0 && valB > 0);
        if (doesCross) {
            float t = valA / change;
            float3 crossingCoord = float3(coordA) + t * float3(coordB - coordA);
            posAccum += crossingCoord;
            numCrossings++;
        }
    }

    if (numCrossings < 3) {
        coordVertMap[flattenCoord(coord, voxCount)] = INVALID_VERTEX;
        return;
    }

    posAccum /= float(numCrossings);
    float3 pos = coordToPos(posAccum, params.voxCount, params.voxSize);
    float3 norm = length(dirAccum) > 1e-6 ? normalize(dirAccum) : float3(0, 1, 0);

    int vertIdx = atomic_fetch_add_explicit(vertexCounter, 1, memory_order_relaxed);
    vertices[vertIdx] = Vertex { pos, norm };
    coordVertMap[flattenCoord(coord, voxCount)] = vertIdx;
}

kernel void surfaceNetsIndices(
    texture3d<float, access::read> volume [[texture(0)]],
    device int* coordVertMap [[buffer(0)]],
    device uint* triangles [[buffer(1)]],
    device atomic_int* triCounter [[buffer(2)]],
    constant MeshParams& params [[buffer(3)]],
    device Vertex* vertices [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int3 voxCount = params.voxCount;
    int3 regionSize = params.regionMax - params.regionMin;
    int regionTotal = regionSize.x * regionSize.y * regionSize.z;
    if (int(tid) >= regionTotal) return;

    int3 localCoord = unflattenCoord(int(tid), regionSize);
    int3 coord = localCoord + params.regionMin;

    int vertA = coordVertMap[flattenCoord(coord, voxCount)];
    if (vertA == INVALID_VERTEX) return;

    // Check 3 axes
    int3 axes[3] = { int3(1,0,0), int3(0,1,0), int3(0,0,1) };
    int3 d1s[3]  = { int3(0,0,1), int3(1,0,0), int3(0,1,0) };
    int3 d2s[3]  = { int3(0,1,0), int3(0,0,1), int3(1,0,0) };

    for (int ax = 0; ax < 3; ax++) {
        int3 axis = axes[ax];
        int3 d1 = d1s[ax];
        int3 d2 = d2s[ax];

        // Need neighbors in d1 and d2 directions
        if (any(coord - d1 < 0) || any(coord - d2 < 0)) continue;

        float valA = readVolume(volume, coord, voxCount);
        float valB = readVolume(volume, coord + axis, voxCount);

        // No crossing on this axis
        if ((valA < 0) == (valB < 0)) continue;

        int a = vertA;
        int b = coordVertMap[flattenCoord(coord - d1, voxCount)];
        int c = coordVertMap[flattenCoord(coord - (d1 + d2), voxCount)];
        int d = coordVertMap[flattenCoord(coord - d2, voxCount)];

        if (b == INVALID_VERTEX || c == INVALID_VERTEX || d == INVALID_VERTEX) continue;

        int triIdx = atomic_fetch_add_explicit(triCounter, 6, memory_order_relaxed);

        if (valA < 0) {
            triangles[triIdx + 0] = uint(c);
            triangles[triIdx + 1] = uint(b);
            triangles[triIdx + 2] = uint(a);
            triangles[triIdx + 3] = uint(d);
            triangles[triIdx + 4] = uint(c);
            triangles[triIdx + 5] = uint(a);
        } else {
            triangles[triIdx + 0] = uint(a);
            triangles[triIdx + 1] = uint(c);
            triangles[triIdx + 2] = uint(d);
            triangles[triIdx + 3] = uint(a);
            triangles[triIdx + 4] = uint(b);
            triangles[triIdx + 5] = uint(c);
        }
    }
}
