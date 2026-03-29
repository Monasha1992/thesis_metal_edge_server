#include <metal_stdlib>
using namespace metal;

kernel void depthStats(
                       texture2d<float, access::read> depthTex [[texture(0)]],
                       device atomic_uint* minVal [[buffer(0)]],
                       device atomic_uint* maxVal [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]
                       )
{
    if (gid.x >= depthTex.get_width() || gid.y >= depthTex.get_height()) return;

    float depth = depthTex.read(gid).r;
    if (depth <= 0.0) return;

    // Store as fixed-point (multiply by 10000) since atomics need integers
    uint depthInt = uint(depth * 10000.0);

    atomic_fetch_min_explicit(minVal, depthInt, memory_order_relaxed);
    atomic_fetch_max_explicit(maxVal, depthInt, memory_order_relaxed);
}
