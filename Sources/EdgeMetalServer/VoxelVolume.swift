import Foundation
import simd

// ── Chunk grid layout ─────────────────────────────────────────────────────────
// Mirrors ChunkManager.cs — divides the TSDF voxel volume into axis-aligned
// sub-regions so each can be read back from the GPU and meshed independently.

struct ChunkCoord: Hashable {
    let x, y, z: Int
}

struct VoxelChunk {
    let coord:      ChunkCoord
    /// World-space lower corner of this chunk (origin for vertex positions).
    let worldPos:   SIMD3<Float>
    /// Inclusive start index inside the global voxel volume.
    let voxelStart: SIMD3<Int>
    /// Number of voxels along each axis (readback size).
    let voxelSize:  SIMD3<Int>
}

struct VoxelVolume {

    // Global volume parameters (mirror DepthProcessor)
    let voxCount:       SIMD3<Int>
    let voxSize:        Float

    // Chunk grid parameters — match ChunkManager.cs defaults
    let chunkWorldSize: Float   // world-space side length of one chunk (default 5 m)
    let overlap:        Float   // extra voxels shared between adjacent chunks (default 0.5 m)

    init(voxCount: SIMD3<Int>, voxSize: Float,
         chunkWorldSize: Float = 5.0,
         overlap:        Float = 0.5) {
        self.voxCount       = voxCount
        self.voxSize        = voxSize
        self.chunkWorldSize = chunkWorldSize
        self.overlap        = overlap
    }

    // ── Volume origin ─────────────────────────────────────────────────────────
    // The Metal kernel places voxel centres at:
    //   worldPos = (indices + 0.5 - voxCount/2) * voxSize
    // So the lower corner (before voxel 0) is at:
    //   origin = -voxCount/2 * voxSize
    var volumeOriginWorld: SIMD3<Float> {
        SIMD3<Float>(
            Float(-voxCount.x) * 0.5 * voxSize,
            Float(-voxCount.y) * 0.5 * voxSize,
            Float(-voxCount.z) * 0.5 * voxSize)
    }

    /// Chunk side length in voxels (rounded up).
    var chunkSizeVox: Int { max(2, Int(ceil(chunkWorldSize / voxSize))) }

    /// Overlap in voxels (rounded up).
    var overlapVox:   Int { max(0, Int(ceil(overlap / voxSize))) }

    // ── Build chunk list ──────────────────────────────────────────────────────
    // Returns all VoxelChunks that tile the full voxel volume, including overlap.

    func allChunks() -> [VoxelChunk] {
        let csv = chunkSizeVox
        let ov  = overlapVox

        let ccx = (voxCount.x + csv - 1) / csv
        let ccy = (voxCount.y + csv - 1) / csv
        let ccz = (voxCount.z + csv - 1) / csv

        var result: [VoxelChunk] = []
        result.reserveCapacity(ccx * ccy * ccz)

        let origin = volumeOriginWorld

        for cz in 0..<ccz {
        for cy in 0..<ccy {
        for cx in 0..<ccx {
            let sx = cx * csv
            let sy = cy * csv
            let sz = cz * csv

            // End index is exclusive; clamp to volume bounds; add overlap
            let ex = min(sx + csv + ov, voxCount.x)
            let ey = min(sy + csv + ov, voxCount.y)
            let ez = min(sz + csv + ov, voxCount.z)

            let size = SIMD3<Int>(ex - sx, ey - sy, ez - sz)
            guard size.x > 2 && size.y > 2 && size.z > 2 else { continue }

            let worldPos = origin + SIMD3<Float>(
                Float(sx) * voxSize,
                Float(sy) * voxSize,
                Float(sz) * voxSize)

            result.append(VoxelChunk(
                coord:      ChunkCoord(x: cx, y: cy, z: cz),
                worldPos:   worldPos,
                voxelStart: SIMD3<Int>(sx, sy, sz),
                voxelSize:  size))
        }}}
        return result
    }
}
