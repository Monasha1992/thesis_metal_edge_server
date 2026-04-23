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

    static func main() {
        let port: UInt16 = 9876

        // Start a TCP server on port 9876
        let listener = try! NWListener(using: .tcp, on: NWEndpoint.Port(rawValue: port)!)

        // Called every time a new Quest connects
        listener.newConnectionHandler = { connection in
            print("Client connected: \(connection.endpoint)")
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

                // Step 8: Run Surface Nets to extract a triangle mesh from the TSDF volume.
                //   Only meshes the region around the camera (not the full 128³ volume).
                let mesh = metal.generateMesh(frame: frame)
                let t7 = Date()

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
                → \(mesh.vertexCount)v \(mesh.triangleCount)t
                """)

                // ── Serialize mesh and send back to Quest ─────────────────────
                //
                // Response payload layout:
                //   8 bytes  — timestamp echo (uint64, same value received from Quest)
                //   4 bytes  — vertex count (uint32)
                //   4 bytes  — index count (uint32)
                //   N×24 bytes — vertices: 6 floats each (pos.xyz + normal.xyz)
                //   M×4 bytes  — triangle indices (uint32 each)
                var payload = Data()

                // Echo the original timestamp so the Quest can compute RTT
                var ts = frame.timestamp
                payload.append(Data(bytes: &ts, count: 8))

                // Vertex and index counts
                var vertCount = UInt32(mesh.vertexCount)
                var idxCount  = UInt32(mesh.indices.count)
                payload.append(Data(bytes: &vertCount, count: 4))
                payload.append(Data(bytes: &idxCount, count: 4))

                // Interleaved vertex data: position XYZ then normal XYZ (24 bytes per vertex)
                var verts = mesh.vertices
                payload.append(Data(bytes: &verts, count: verts.count * 4))

                // Triangle indices — every 3 uint32 indices = one triangle
                var indices = mesh.indices
                payload.append(Data(bytes: &indices, count: indices.count * 4))

                // 5-byte header: [type=0x03][payload length as big-endian uint32]
                var response = Data()
                response.append(0x03)
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
}
