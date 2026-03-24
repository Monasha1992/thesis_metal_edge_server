import Foundation
import simd

// ── Surface-nets mesher ───────────────────────────────────────────────────────
// Direct port of NetMesher.cs (VertexJob + IndexJob) from the Unity project.
//
// Input:  [Float] voxel volume in [-1, 1], where -1.0 exactly == EMPTY_VOXEL.
//         Values are read back from the Metal R32Float 3D texture via
//         DepthProcessor.readbackVolumeRegion().
// Output: MeshOutput — vertices, normals, and indices for one MeshChunkPacket.
//
// Algorithm: Dual contouring / surface nets.
//   VertexJob — for each interior voxel, finds zero-crossings across the 12 cube
//               edges, averages their positions, and emits one vertex.
//   IndexJob  — for each vertex, quads the three axis-aligned face pairs.

struct MeshOutput {
    var vertices: [SIMD3<Float>]
    var normals:  [SIMD3<Float>]
    var indices:  [Int32]
}

enum MeshGenerator {

    // Sentinel values (mirrors Mesher.cs constants)
    private static let emptyVoxel:   Float = -1.0
    private static let invalidVert:  Int   = -1

    // 8 corners of a unit cube, indexed [0..7]
    // Row-0: bottom face (y=0), clockwise from front-left
    // Row-1: top face    (y=1)
    private static let cornerOffs: [(Int, Int, Int)] = [
        (0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1),
        (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)
    ]

    // 12 edges: each edge is a pair of corner indices
    private static let crnrIdxA: [Int] = [0, 1, 2, 3,  4, 5, 6, 7,  0, 1, 2, 3]
    private static let crnrIdxB: [Int] = [1, 2, 3, 0,  5, 6, 7, 4,  4, 5, 6, 7]

    // ── Public entry point ────────────────────────────────────────────────────

    static func generate(volume: [Float],
                         voxCount: SIMD3<Int>,
                         voxSize: Float) -> MeshOutput {

        let total = voxCount.x * voxCount.y * voxCount.z
        guard total > 0, volume.count >= total else {
            return MeshOutput(vertices: [], normals: [], indices: [])
        }

        // Vertex pass
        var vertices   = [SIMD3<Float>]()
        var normals    = [SIMD3<Float>]()
        var vertCoords = [SIMD3<Int>]()
        var coordVertMap = [Int](repeating: invalidVert, count: total)

        vertices.reserveCapacity(total / 4)
        vertCoords.reserveCapacity(total / 4)

        for i in 0..<total {
            let c = indexToCoord(i, s: voxCount)

            // Skip border voxels (need neighbours in all directions for quads)
            if c.x == voxCount.x - 1 ||
               c.y == voxCount.y - 1 ||
               c.z == voxCount.z - 1 {
                coordVertMap[i] = invalidVert
                continue
            }

            var posCoord = SIMD3<Float>(repeating: 0)
            var dir      = SIMD3<Float>(repeating: 0)
            var numCrossings    = 0
            var numBadCrossings = 0

            for e in 0..<12 {
                let coA = cornerOffs[crnrIdxA[e]]
                let coB = cornerOffs[crnrIdxB[e]]

                let cA = SIMD3<Int>(c.x + coA.0, c.y + coA.1, c.z + coA.2)
                let cB = SIMD3<Int>(c.x + coB.0, c.y + coB.1, c.z + coB.2)

                let rawA = rawAt(cA, volume: volume, s: voxCount)
                let rawB = rawAt(cB, volume: volume, s: voxCount)

                // Map empty → 0 for crossing / gradient computation
                let valA = rawA == emptyVoxel ? 0.0 : rawA
                let valB = rawB == emptyVoxel ? 0.0 : rawB

                let change = valA - valB

                // Gradient contribution (same as `dir += float3(coordA - coordB) * change`)
                let diff = SIMD3<Float>(
                    Float(cA.x - cB.x),
                    Float(cA.y - cB.y),
                    Float(cA.z - cB.z))
                dir += diff * change

                let doesCross = (valA < 0) != (valB < 0)
                if doesCross {
                    if rawA == emptyVoxel || rawB == emptyVoxel {
                        numBadCrossings += 1
                    }
                    // Linear interpolation to find crossing coordinate
                    let t = valA / change
                    let crossCoord = SIMD3<Float>(
                        Float(cA.x) + t * Float(cB.x - cA.x),
                        Float(cA.y) + t * Float(cB.y - cA.y),
                        Float(cA.z) + t * Float(cB.z - cA.z))
                    posCoord += crossCoord
                    numCrossings += 1
                }
            }

            // Require at least 3 valid crossings, reject if all are bad (empty boundaries)
            if numCrossings < 3 || numCrossings == numBadCrossings {
                coordVertMap[i] = invalidVert
                continue
            }

            posCoord /= Float(numCrossings)

            // CoordToPos: c * voxSize + voxSize * 0.5  (centre of voxel at posCoord)
            let pos  = posCoord * voxSize + voxSize * 0.5
            let len  = simd_length(dir)
            let norm = len > 1e-8 ? dir / len : SIMD3<Float>(0, 1, 0)

            coordVertMap[i] = vertices.count
            vertices.append(pos)
            normals.append(norm)
            vertCoords.append(c)
        }

        guard vertices.count >= 3 else {
            return MeshOutput(vertices: [], normals: [], indices: [])
        }

        // Index pass
        var indices = [Int32]()
        indices.reserveCapacity(vertices.count * 6)

        // IndexJob calls TrisForAxis for (axis, d1, d2):
        //   X-axis: d1=Z, d2=Y
        //   Y-axis: d1=X, d2=Z
        //   Z-axis: d1=Y, d2=X
        let axisX = SIMD3<Int>(1, 0, 0)
        let axisY = SIMD3<Int>(0, 1, 0)
        let axisZ = SIMD3<Int>(0, 0, 1)

        for coord in vertCoords {
            addTris(coord, axis: axisX, d1: axisZ, d2: axisY,
                    volume: volume, s: voxCount, map: coordVertMap, out: &indices)
            addTris(coord, axis: axisY, d1: axisX, d2: axisZ,
                    volume: volume, s: voxCount, map: coordVertMap, out: &indices)
            addTris(coord, axis: axisZ, d1: axisY, d2: axisX,
                    volume: volume, s: voxCount, map: coordVertMap, out: &indices)
        }

        return MeshOutput(vertices: vertices, normals: normals, indices: indices)
    }

    // ── Index quad emission ───────────────────────────────────────────────────
    // Port of TrisForAxis() in IndexJob.
    // Note: IndexJob does NOT remap empty→0, so empty voxels are treated as
    // negative (inside) for the sign-change test, matching the original.

    private static func addTris(
        _ coord: SIMD3<Int>,
        axis: SIMD3<Int>, d1: SIMD3<Int>, d2: SIMD3<Int>,
        volume: [Float], s: SIMD3<Int>,
        map: [Int], out: inout [Int32])
    {
        // Bounds check: coord - d1 and coord - d2 must be >= 0
        let cd1 = SIMD3<Int>(coord.x - d1.x, coord.y - d1.y, coord.z - d1.z)
        let cd2 = SIMD3<Int>(coord.x - d2.x, coord.y - d2.y, coord.z - d2.z)
        if cd1.x < 0 || cd1.y < 0 || cd1.z < 0 { return }
        if cd2.x < 0 || cd2.y < 0 || cd2.z < 0 { return }

        let va = volume[flatCoord(coord, s: s)]
        let vb = volume[flatCoord(SIMD3<Int>(coord.x + axis.x,
                                             coord.y + axis.y,
                                             coord.z + axis.z), s: s)]

        guard (va < 0) != (vb < 0) else { return }   // no sign change → no quad

        let a = map[flatCoord(coord,                                                s: s)]
        let b = map[flatCoord(cd1,                                                  s: s)]
        let c = map[flatCoord(SIMD3<Int>(coord.x - d1.x - d2.x,
                                         coord.y - d1.y - d2.y,
                                         coord.z - d1.z - d2.z),                   s: s)]
        let d = map[flatCoord(cd2,                                                  s: s)]

        guard a != invalidVert, b != invalidVert,
              c != invalidVert, d != invalidVert else { return }

        if va < 0 {
            // Inside → winding: c, b, a / d, c, a
            out.append(Int32(c)); out.append(Int32(b)); out.append(Int32(a))
            out.append(Int32(d)); out.append(Int32(c)); out.append(Int32(a))
        } else {
            // Outside → winding: a, c, d / a, b, c
            out.append(Int32(a)); out.append(Int32(c)); out.append(Int32(d))
            out.append(Int32(a)); out.append(Int32(b)); out.append(Int32(c))
        }
    }

    // ── Coordinate helpers ────────────────────────────────────────────────────

    @inline(__always)
    private static func flatCoord(_ c: SIMD3<Int>, s: SIMD3<Int>) -> Int {
        c.x + c.y * s.x + c.z * s.x * s.y
    }

    @inline(__always)
    private static func indexToCoord(_ i: Int, s: SIMD3<Int>) -> SIMD3<Int> {
        SIMD3<Int>(
            i % s.x,
            (i / s.x) % s.y,
            i / (s.x * s.y))
    }

    /// Raw voxel value — may be -1.0 (EMPTY_VOXEL).
    @inline(__always)
    private static func rawAt(_ c: SIMD3<Int>, volume: [Float], s: SIMD3<Int>) -> Float {
        volume[flatCoord(c, s: s)]
    }
}
