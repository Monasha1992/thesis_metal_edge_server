import Foundation
import Network

// ─────────────────────────────────────────────────────────────────────────────
// EdgeMetalServer — Entry point for the Mac edge server
//
// WHAT THIS FILE DOES:
//   This is the main program that runs on the Mac. It listens for a connection
//   from the Quest over Wi-Fi, receives raw depth frames, runs the full GPU
//   processing pipeline on the Mac, and sends the resulting triangle mesh back.
//
// HOW IT FITS IN THE SYSTEM:
//   Quest 3  ──(raw depth + matrices)──►  Mac (this server)
//   Quest 3  ◄─────(triangle mesh)──────  Mac (this server)
//
// MESSAGE PROTOCOL (5-byte header + payload):
//   Outgoing (Quest → Mac):
//     Byte 0      = message type (0x01 = depth frame)
//     Bytes 1–4   = payload length as big-endian uint32
//     Bytes 5–12  = uint64 timestamp (ms since epoch, for latency measurement)
//     Bytes 13+   = frame data (matrices + volume config + depth pixels)
//
//   Incoming (Mac → Quest):
//     Byte 0      = message type (0x03 = triangle mesh)
//     Bytes 1–4   = payload length as big-endian uint32
//     Bytes 5–12  = uint64 timestamp echo (same value from incoming frame)
//     Bytes 13+   = mesh data (vertex count + index count + vertices + indices)
// ─────────────────────────────────────────────────────────────────────────────

@main
struct EdgeMetalServer {

    // Shared Metal GPU pipeline — created once, reused for every frame
    static let metal = MetalPipeline()

    // ── Mesh-delivery mode ────────────────────────────────────────────────────
    // true  → chunked meshing (0x04): the volume is meshed along a fixed grid
    //         of 32³-voxel chunks; up to `maxChunksPerFrame` camera-near chunks
    //         ship per response, each tagged with its grid coordinate. The
    //         Quest caches chunks in a dictionary so previously-seen geometry
    //         persists when the camera looks away (standalone-style).
    // false → legacy single-mesh (0x03): one camera-centred region mesh that
    //         the Quest replaces wholesale each round (no persistence).
    //
    // Both ends support both message types, so this flag is the only switch —
    // useful as a thesis ablation (chunked persistence vs global replace).
    static let useChunkedMeshing = true

    // How many chunks to mesh + send per incoming depth frame. ~30 chunks are
    // in range of a 6 m maxUpdateDist, so 8/frame refreshes every visible
    // chunk roughly every 4 frames (~2.5 Hz per chunk at a 10 Hz send rate).
    static let maxChunksPerFrame = 8

    static func main() {
        let port: UInt16 = 9876

        // Start a TCP server on port 9876
        let listener = try! NWListener(using: .tcp, on: NWEndpoint.Port(rawValue: port)!)

        // Called every time a new Quest connects
        listener.newConnectionHandler = { connection in
            print("Client connected: \(connection.endpoint)")

            // Open a fresh metrics CSV for this connection. The per-frame
            // recorder appends rows to it as receivePayload processes each
            // depth frame; the file is closed on process exit. See
            // MetricsRecorder.swift for the format and join semantics.
            MetricsRecorder.shared.startSession()

            connection.start(queue: .main)
            Self.receive(on: connection)
        }

        listener.start(queue: .main)
        print("Server listening on port \(port)")

        // Keep the server running forever
        dispatchMain()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // receive — Reads the 5-byte message header from the Quest
    //
    // The header tells us:
    //   - what type of message is coming (byte 0)
    //   - how many bytes the payload is (bytes 1-4)
    //
    // After reading the header we hand off to receivePayload() to get the rest.
    // ─────────────────────────────────────────────────────────────────────────
    static func receive(on connection: NWConnection) {
        // Read exactly 5 bytes — the fixed-size message header
        connection.receive(minimumIncompleteLength: 5, maximumLength: 5) { data, _, _, error in
            if let error = error {
                print("Error: \(error)")
                return
            }

            guard let data = data, data.count == 5 else {
                print("Connection closed")
                return
            }

            // Parse header: first byte = type, next 4 bytes = payload length (big-endian)
            let type   = data[0]
            let length = Int(data[1]) << 24 | Int(data[2]) << 16 | Int(data[3]) << 8 | Int(data[4])

            // Now read the full payload (may arrive in multiple chunks)
            receivePayload(on: connection, type: type, remaining: length, accumulated: Data())
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // receivePayload — Accumulates the full message payload, then processes it
    //
    // TCP can split a large message across multiple packets, so we keep reading
    // until we have all the bytes we expect. Once complete, we run the full
    // GPU pipeline and send the mesh back.
    //
    // PIPELINE (all on Mac GPU):
    //   1.  Parse the raw bytes into a DepthFrame struct
    //   2.  Upload depth pixels to a Metal GPU texture
    //   3.  Dilate the depth (fills holes from sensor noise)
    //   4.  Estimate surface normals from the depth image
    //   5.  Set up the 3D TSDF volume (done once, persists across frames)
    //   6.  Set up the frustum point list (done once, persists across frames)
    //   7.  Integrate the depth frame into the TSDF volume
    //   8.  Run Surface Nets to extract a triangle mesh from the volume
    //   9.  Serialize the mesh (with echoed timestamp) and send back to Quest
    //
    // TIMING:
    //   Each step is timed and printed so you can see where time is spent.
    //   "wall" = real elapsed time from frame arrival to mesh sent.
    // ─────────────────────────────────────────────────────────────────────────
    static func receivePayload(
        on connection: NWConnection, type: UInt8, remaining: Int, accumulated: Data
    ) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: remaining) {
            data, _, _, error in
            if let error = error {
                print("Error: \(error)")
                return
            }

            guard let data = data else { return }

            var accumulated = accumulated
            accumulated.append(data)
            let left = remaining - data.count

            if left > 0 {
                // Haven't received the full message yet — keep reading
                receivePayload(
                    on: connection, type: type, remaining: left, accumulated: accumulated)
            } else {
                // ── Full message received — run the GPU pipeline ──────────────
                let wallStart = Date()

                // Step 1: Parse raw bytes into a structured DepthFrame.
                //   First 8 bytes = latency timestamp (we echo this back to Quest).
                //   Next 540 bytes = camera matrices + volume config.
                //   Rest = raw float32 depth pixels.
                let frame = parseDepthFrame(accumulated)
                let t1 = Date()

                // Step 2: Upload the raw float32 depth pixels to a Metal GPU texture
                let depthTexture = metal.uploadDepthTexture(frame: frame)
                let t2 = Date()

                // Step 3: Dilate the depth image — fills holes caused by reflections
                //   or sensor noise using 8 passes of jump-flood morphological dilation.
                let dilatedDepth = metal.dilateDepth(depthTexture: depthTexture, frame: frame)
                let t3 = Date()

                // Step 4: Estimate surface normals from the original depth image.
                //   Each pixel gets a direction vector showing which way that surface faces.
                //   We use the undilated depth here for cleaner edges.
                let normalsTexture = metal.generateNormals(depthTexture: depthTexture, frame: frame)
                let t4 = Date()

                // Step 5: Create the 3D TSDF volume texture (only happens on first frame).
                //   The volume persists — it accumulates observations across many frames.
                metal.setupVolume(frame: frame)

                // Step 6: Pre-compute the list of voxel positions inside the camera frustum.
                //   (only happens on first frame — the frustum shape stays constant)
                metal.setupFrustum(frame: frame)
                let t5 = Date()

                // Step 7: Integrate this depth frame into the persistent TSDF volume.
                //   Each voxel stores: how far is it from the nearest surface?
                //   Observations are blended using a weighted running average (weight capped at 30).
                metal.integrateDepth(
                    depthTexture: depthTexture,
                    normTexture:  normalsTexture,
                    dilatedDepth: dilatedDepth,
                    frame: frame
                )
                let t6 = Date()

                // Step 8: Run Surface Nets to extract triangles from the TSDF volume.
                //   Chunked mode: mesh up to maxChunksPerFrame grid chunks near the
                //                 camera (round-robin through the in-range set).
                //   Legacy mode:  one camera-centred region mesh.
                let messageType: UInt8
                let payload: Data
                let totalVerts: Int
                let totalTris:  Int

                let chunks: [MetalPipeline.ChunkMesh]
                let singleMesh: MetalPipeline.MeshResult?

                if useChunkedMeshing {
                    chunks     = metal.generateChunkMeshes(frame: frame, budget: maxChunksPerFrame)
                    singleMesh = nil
                } else {
                    chunks     = []
                    singleMesh = metal.generateMesh(frame: frame)
                }
                let t7 = Date()

                if useChunkedMeshing {
                    messageType = 0x04
                    payload     = serializeChunkBatch(chunks, timestamp: frame.timestamp,
                                                      totalMs: Float(t7.timeIntervalSince(wallStart) * 1000),
                                                      parseMs: Float(t1.timeIntervalSince(wallStart) * 1000),
                                                      integrateMs: Float(t6.timeIntervalSince(t5) * 1000),
                                                      meshMs: Float(t7.timeIntervalSince(t6) * 1000))
                    totalVerts  = chunks.reduce(0) { $0 + $1.mesh.vertexCount }
                    totalTris   = chunks.reduce(0) { $0 + $1.mesh.triangleCount }
                } else {
                    let mesh    = singleMesh!
                    messageType = 0x03
                    payload     = serializeSingleMesh(mesh, timestamp: frame.timestamp,
                                                      totalMs: Float(t7.timeIntervalSince(wallStart) * 1000),
                                                      parseMs: Float(t1.timeIntervalSince(wallStart) * 1000),
                                                      integrateMs: Float(t6.timeIntervalSince(t5) * 1000),
                                                      meshMs: Float(t7.timeIntervalSince(t6) * 1000))
                    totalVerts  = mesh.vertexCount
                    totalTris   = mesh.triangleCount
                }

                // ── Per-step timing breakdown (printed for performance analysis) ──
                func ms(_ a: Date, _ b: Date) -> String {
                    String(format: "%.1f", b.timeIntervalSince(a) * 1000)
                }
                let wallMs = String(format: "%.1f", t7.timeIntervalSince(wallStart) * 1000)
                print("""
                [frame] parse=\(ms(wallStart,t1))ms upload=\(ms(t1,t2))ms \
                dilate=\(ms(t2,t3))ms normals=\(ms(t3,t4))ms \
                setup=\(ms(t4,t5))ms integrate=\(ms(t5,t6))ms \
                mesh=\(ms(t6,t7))ms | total=\(wallMs)ms \
                → \(totalVerts)v \(totalTris)t
                """)

                // ── Append a CSV row for the metrics-recorder ────────────────
                // The Quest side stamps and echoes a uint64 timestamp; we use
                // it as the join key when merging this Mac CSV with the
                // Quest's `mesh` rows in pandas. Bytes-in/out include the
                // 5-byte framing header on each direction (see [04-protocol]).
                MetricsRecorder.shared.record(
                    timestampEchoMs:  frame.timestamp,
                    parseMs:          t1.timeIntervalSince(wallStart) * 1000,
                    uploadMs:         t2.timeIntervalSince(t1)        * 1000,
                    dilateMs:         t3.timeIntervalSince(t2)        * 1000,
                    normalsMs:        t4.timeIntervalSince(t3)        * 1000,
                    setupMs:          t5.timeIntervalSince(t4)        * 1000,
                    integrateMs:      t6.timeIntervalSince(t5)        * 1000,
                    meshMs:           t7.timeIntervalSince(t6)        * 1000,
                    totalMs:          t7.timeIntervalSince(wallStart) * 1000,
                    vertCount:        totalVerts,
                    triCount:         totalTris,
                    payloadInBytes:   accumulated.count + 5,
                    responseOutBytes: 5 + payload.count
                )

                // ── Frame and send ────────────────────────────────────────────
                // 5-byte header: [type][payload length as big-endian uint32]
                var response = Data()
                response.append(messageType)
                let len = UInt32(payload.count)
                response.append(contentsOf: [
                    UInt8((len >> 24) & 0xFF),
                    UInt8((len >> 16) & 0xFF),
                    UInt8((len >> 8)  & 0xFF),
                    UInt8( len        & 0xFF)
                ])
                response.append(payload)

                connection.send(content: response, completion: .contentProcessed { _ in
                    print("[sent] \(response.count) bytes")
                })

                // Ready to receive the next depth frame
                receive(on: connection)
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // serializeSingleMesh — Legacy 0x03 payload (one global mesh)
    //
    // Layout:
    //   8 bytes    — timestamp echo (uint64, same value received from Quest)
    //   4 bytes    — vertex count (uint32)
    //   4 bytes    — index count (uint32)
    //   N×24 bytes — vertices: 6 floats each (pos.xyz + normal.xyz)
    //   M×4 bytes  — triangle indices (uint32 each)
    // ─────────────────────────────────────────────────────────────────────────
    static func serializeSingleMesh(_ mesh: MetalPipeline.MeshResult, timestamp: UInt64, totalMs: Float, parseMs: Float, integrateMs: Float, meshMs: Float) -> Data {
        var payload = Data()

        var ts = timestamp
        payload.append(Data(bytes: &ts, count: 8))

        var t0 = totalMs;     payload.append(Data(bytes: &t0, count: 4))
        var t1 = parseMs;     payload.append(Data(bytes: &t1, count: 4))
        var t2 = integrateMs; payload.append(Data(bytes: &t2, count: 4))
        var t3 = meshMs;      payload.append(Data(bytes: &t3, count: 4))

        var vertCount = UInt32(mesh.vertexCount)
        var idxCount  = UInt32(mesh.indices.count)
        payload.append(Data(bytes: &vertCount, count: 4))
        payload.append(Data(bytes: &idxCount, count: 4))

        var verts = mesh.vertices
        payload.append(Data(bytes: &verts, count: verts.count * 4))

        var indices = mesh.indices
        payload.append(Data(bytes: &indices, count: indices.count * 4))

        return payload
    }

    // ─────────────────────────────────────────────────────────────────────────
    // serializeChunkBatch — 0x04 payload (persistent chunk cache protocol)
    //
    // Layout:
    //   8 bytes  — timestamp echo (uint64, same value received from Quest)
    //   4 bytes  — chunk count (uint32)
    //   then per chunk:
    //     12 bytes   — chunk grid coordinate (3 × int32)
    //     4 bytes    — vertex count (uint32)
    //     4 bytes    — index count (uint32)
    //     N×24 bytes — vertices (pos.xyz + normal.xyz, float32 each)
    //     M×4 bytes  — triangle indices (uint32, LOCAL to this chunk's vertices)
    //
    //   A chunk with vertexCount == 0 means "this grid cell is now empty" —
    //   the Quest clears its cached mesh for that cell.
    // ─────────────────────────────────────────────────────────────────────────
    static func serializeChunkBatch(_ chunks: [MetalPipeline.ChunkMesh], timestamp: UInt64, totalMs: Float, parseMs: Float, integrateMs: Float, meshMs: Float) -> Data {
        var payload = Data()

        var ts = timestamp
        payload.append(Data(bytes: &ts, count: 8))

        var t0 = totalMs;     payload.append(Data(bytes: &t0, count: 4))
        var t1 = parseMs;     payload.append(Data(bytes: &t1, count: 4))
        var t2 = integrateMs; payload.append(Data(bytes: &t2, count: 4))
        var t3 = meshMs;      payload.append(Data(bytes: &t3, count: 4))

        var count = UInt32(chunks.count)
        payload.append(Data(bytes: &count, count: 4))

        for chunk in chunks {
            var cx = chunk.coord.x
            var cy = chunk.coord.y
            var cz = chunk.coord.z
            payload.append(Data(bytes: &cx, count: 4))
            payload.append(Data(bytes: &cy, count: 4))
            payload.append(Data(bytes: &cz, count: 4))

            var vertCount = UInt32(chunk.mesh.vertexCount)
            var idxCount  = UInt32(chunk.mesh.indices.count)
            payload.append(Data(bytes: &vertCount, count: 4))
            payload.append(Data(bytes: &idxCount, count: 4))

            var verts = chunk.mesh.vertices
            payload.append(Data(bytes: &verts, count: verts.count * 4))

            var indices = chunk.mesh.indices
            payload.append(Data(bytes: &indices, count: indices.count * 4))
        }

        return payload
    }
}
