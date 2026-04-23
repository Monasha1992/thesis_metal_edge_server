import Foundation
import simd

// ─────────────────────────────────────────────────────────────────────────────
// DepthFrame.swift — The deserialized form of one incoming depth packet
//
// The Quest sends a binary payload structured as:
//
//   Bytes 0–7     : uint64 timestamp (milliseconds since Unix epoch)
//                   Used for round-trip latency measurement — the Mac echoes
//                   this value back in the mesh response header.
//
//   Bytes 8–519   : 8 float4x4 camera matrices (64 bytes each)
//                   view[0],    view[1]      — world → camera  (left, right eye)
//                   proj[0],    proj[1]      — camera → clip space
//                   viewInv[0], viewInv[1]   — camera → world space
//                   projInv[0], projInv[1]   — clip → camera space
//
//   Bytes 520–547 : volume configuration (7 × 4 bytes = 28 bytes)
//                   voxelCount.xyz (3× int32)
//                   voxelSize      (float32)
//                   voxelDistance  (float32)
//                   maxUpdateDist  (float32)
//                   numPlayers     (int32) — how many of the 8 slots below are valid
//
//   Bytes 548–643 : player head world positions (8 slots × 3 floats = 96 bytes)
//                   Each slot is (x, y, z) as float32.
//                   Only the first `numPlayers` slots hold real positions.
//                   Remaining slots are zero and ignored by the integration shader.
//
//   Bytes 644+    : raw depth pixels (float32 per pixel, 320 pixels wide)
// ─────────────────────────────────────────────────────────────────────────────

// Fixed maximum number of player head slots sent per frame.
// Must match MAX_PLAYERS in EdgeServerClient.cs and the playerHeads[] array
// size in VolumeIntegration.metal.
let MAX_PLAYERS: Int = 8

struct DepthFrame {
    // ── Camera matrices (2 eyes each) ────────────────────────────────────────
    let view:    [float4x4]   // world → camera space
    let proj:    [float4x4]   // camera → clip space (perspective)
    let viewInv: [float4x4]   // camera → world space (column 3 = eye world position)
    let projInv: [float4x4]   // clip → camera space (used for unprojection)

    // ── Volume configuration ──────────────────────────────────────────────────
    let voxelCount:    SIMD3<Int32>  // Grid dimensions, e.g. (128, 128, 128)
    let voxelSize:     Float          // Metres per voxel, e.g. 0.1
    let voxelDist:     Float          // TSDF truncation distance, e.g. 0.2
    let maxUpdateDist: Float          // Integration depth limit in metres
    let numPlayers:    Int32          // How many of playerHeads below are valid

    // ── Player head exclusion ─────────────────────────────────────────────────
    // Fixed-size array of 8 world-space head positions.
    // Only the first `numPlayers` entries are valid — the rest are zero.
    // The integrate kernel skips voxels inside a cylinder around each valid head
    // so player bodies don't get baked into the reconstructed room mesh.
    let playerHeads: [SIMD3<Float>]

    // ── Raw depth image ───────────────────────────────────────────────────────
    let depthPixels: Data  // float32 array, row-major, width=320
    let width:       Int   // Hardcoded 320 (Quest depth sensor resolution)
    let height:      Int   // Computed from pixel count

    // ── Latency timestamp ─────────────────────────────────────────────────────
    // Milliseconds since Unix epoch, sent by Quest at capture time.
    // The Mac echoes this value back in the mesh response so the Quest can
    // compute the full round-trip latency when the mesh arrives.
    let timestamp: UInt64
}

// ─────────────────────────────────────────────────────────────────────────────
// parseDepthFrame — Deserializes a raw binary payload into a DepthFrame struct
//
// Must match the serialization layout in EdgeServerClient.cs: SerializeFrameData()
// and the timestamp prepend in OnDepthUpdated().
// ─────────────────────────────────────────────────────────────────────────────
func parseDepthFrame(_ data: Data) -> DepthFrame {
    var offset = 0

    // ── Helper readers (little-endian — matches Unity's BitConverter default) ──
    func readFloat() -> Float {
        var val: Float = 0
        _ = withUnsafeMutableBytes(of: &val) { data.copyBytes(to: $0, from: offset..<offset+4) }
        offset += 4
        return val
    }

    func readInt32() -> Int32 {
        var val: Int32 = 0
        _ = withUnsafeMutableBytes(of: &val) { data.copyBytes(to: $0, from: offset..<offset+4) }
        offset += 4
        return val
    }

    func readUInt64() -> UInt64 {
        var val: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &val) { data.copyBytes(to: $0, from: offset..<offset+8) }
        offset += 8
        return val
    }

    // Reads a 4×4 matrix — Unity serializes column-major, which matches Metal's float4x4
    func readMatrix() -> float4x4 {
        var cols = [SIMD4<Float>](repeating: .zero, count: 4)
        for col in 0..<4 {
            let x = readFloat()
            let y = readFloat()
            let z = readFloat()
            let w = readFloat()
            cols[col] = SIMD4<Float>(x, y, z, w)
        }
        return float4x4(cols)
    }

    // ── Parse in the exact order EdgeServerClient.cs writes them ─────────────

    // 8 bytes: latency timestamp
    let timestamp = readUInt64()

    // 512 bytes: 8 matrices (view/proj/viewInv/projInv × 2 eyes)
    let view    = [readMatrix(), readMatrix()]
    let proj    = [readMatrix(), readMatrix()]
    let viewInv = [readMatrix(), readMatrix()]
    let projInv = [readMatrix(), readMatrix()]

    // 28 bytes: volume config
    let voxelCount    = SIMD3<Int32>(readInt32(), readInt32(), readInt32())
    let voxelSize     = readFloat()
    let voxelDist     = readFloat()
    let maxUpdateDist = readFloat()
    let numPlayers    = readInt32()

    // 96 bytes: 8 player head positions (3 floats each).
    // The Quest always sends 8 slots; only the first `numPlayers` are valid.
    var playerHeads: [SIMD3<Float>] = []
    playerHeads.reserveCapacity(MAX_PLAYERS)
    for _ in 0..<MAX_PLAYERS {
        let x = readFloat()
        let y = readFloat()
        let z = readFloat()
        playerHeads.append(SIMD3<Float>(x, y, z))
    }

    // Remainder: raw float32 depth pixels
    let depthPixels = data.subdata(in: offset..<data.count)
    let pixelCount  = depthPixels.count / 4
    let width       = 320
    let height      = pixelCount / width

    return DepthFrame(
        view: view, proj: proj,
        viewInv: viewInv, projInv: projInv,
        voxelCount:    voxelCount,
        voxelSize:     voxelSize,
        voxelDist:     voxelDist,
        maxUpdateDist: maxUpdateDist,
        numPlayers:    numPlayers,
        playerHeads:   playerHeads,
        depthPixels:   depthPixels,
        width: width,  height: height,
        timestamp:     timestamp
    )
}
