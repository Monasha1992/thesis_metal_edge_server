import Foundation
import Metal
import simd

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Current Unix time in milliseconds.
func currentTimeMs() -> Int64 {
    Int64(Date().timeIntervalSince1970 * 1000)
}

/// Parse a named CLI argument: --key=value → value.
func cliArg(_ key: String, default def: String) -> String {
    CommandLine.arguments.dropFirst()
        .first { $0.hasPrefix("--\(key)=") }
        .map { String($0.dropFirst(key.count + 3)) } ?? def
}

// ── Configuration ─────────────────────────────────────────────────────────────

let logLevelStr   =         cliArg("loglevel",   default: "info")
let port          = UInt16( cliArg("port",        default: "9555"))!
let metricsPath   =         cliArg("metrics",     default: "edge_metrics.csv")
let voxCountArg   = Int(    cliArg("voxcount",    default: "128"))!
let voxSizeArg    = Float(  cliArg("voxsize",     default: "0.1"))!
let maxDistArg    = Float(  cliArg("maxdist",     default: "6.0"))!
let chunkSizeArg  = Float(  cliArg("chunksize",   default: "5.0"))!

// Apply log level before any log() call
currentLogLevel = {
    switch logLevelStr.lowercased() {
    case "debug": return .debug
    case "warn":  return .warn
    case "error": return .error
    default:      return .info
    }
}()

log(.info, "main", """
EdgeMetalServer starting
  port          = \(port)
  loglevel      = \(logLevelStr)
  metrics       = \(metricsPath)
  voxCount      = \(voxCountArg)^3  (\(voxCountArg*voxCountArg*voxCountArg) voxels)
  voxSize       = \(voxSizeArg) m
  maxUpdateDist = \(maxDistArg) m
  chunkSize     = \(chunkSizeArg) m
""")

// ── Metal device ──────────────────────────────────────────────────────────────

guard let device = MTLCreateSystemDefaultDevice() else {
    log(.error, "main", "No Metal device found — is this a Mac with Apple Silicon?")
    exit(1)
}
log(.info, "main", "Metal device: \(device.name)  " +
    "recommendedMaxWorkingSetSize=\(device.recommendedMaxWorkingSetSize / (1024*1024)) MB")

// ── DepthProcessor ────────────────────────────────────────────────────────────

log(.info, "main", "Initialising DepthProcessor …")
let processor: DepthProcessor
do {
    processor = try DepthProcessor(
        device:             device,
        voxCount:           SIMD3<Int>(repeating: voxCountArg),
        voxelSize:          voxSizeArg,
        voxelDist:          voxSizeArg * 2,
        voxelMin:           voxSizeArg,
        maxUpdateDist:      maxDistArg,
        minUpdateDist:      0.5)
} catch {
    log(.error, "main", "DepthProcessor init failed: \(error)")
    exit(1)
}
log(.info, "main", "DepthProcessor ready")

// ── Voxel / chunk layout ──────────────────────────────────────────────────────

let voxelVolume = VoxelVolume(
    voxCount:       processor.voxCount,
    voxSize:        processor.voxelSize,
    chunkWorldSize: chunkSizeArg,
    overlap:        0.5)

let allChunks = voxelVolume.allChunks()
log(.info, "main", "Chunk grid: \(allChunks.count) chunks  " +
    "(chunkSizeVox=\(voxelVolume.chunkSizeVox), overlapVox=\(voxelVolume.overlapVox))")

// ── MetricsLogger ─────────────────────────────────────────────────────────────

let logger: MetricsLogger? = {
    do {
        let l = try MetricsLogger(path: metricsPath)
        log(.info, "main", "Metrics CSV → \(metricsPath)")
        return l
    } catch {
        log(.warn, "main", "Cannot open metrics file '\(metricsPath)': \(error)  — continuing without CSV logging")
        return nil
    }
}()

// ── TCP server ────────────────────────────────────────────────────────────────

let server = TCPServer(port: port)

server.onClientConnected = { clientFD in
    log(.info, "client", "Session started  fd=\(clientFD)")

    var frameCount      = 0
    var totalBytesSent  = 0
    var totalChunks     = 0

    defer {
        log(.info, "client", "Session ended  fd=\(clientFD)  " +
            "frames=\(frameCount)  totalChunks=\(totalChunks)  " +
            "totalKBSent=\(String(format: "%.1f", Double(totalBytesSent)/1024))")
    }

    while true {

        // ── Receive depth frame ───────────────────────────────────────────────
        log(.debug, "net", "Waiting for next packet …")
        guard let raw = readPacket(fd: clientFD) else {
            log(.info, "client", "Connection closed (EOF or recv error)")
            break
        }

        log(.debug, "net", "Received packet  bytes=\(raw.count)  " +
            "type=0x\(raw.isEmpty ? "??" : String(format: "%02X", raw[0]))")

        guard !raw.isEmpty else {
            log(.warn, "net", "Empty packet received, skipping")
            continue
        }
        guard raw[0] == PacketType.depthFrame.rawValue else {
            log(.warn, "net", "Unexpected packet type 0x\(String(format: "%02X", raw[0]))  " +
                "(expected 0x\(String(format: "%02X", PacketType.depthFrame.rawValue))), skipping")
            continue
        }
        guard let frame = parseDepthFrame(raw) else {
            log(.warn, "net", "Failed to parse depth frame  rawBytes=\(raw.count)")
            continue
        }

        frameCount += 1
        let recvMs = currentTimeMs()
        let queueLatencyMs = recvMs - frame.timestampMs
        log(.debug, "client", String(format:
            "Frame #%d  ts=%lldms  size=%d×%d  rawBytes=%d  queueLatency=%lldms",
            frameCount, frame.timestampMs, frame.width, frame.height,
            raw.count, queueLatencyMs))

        let computeStartMs = currentTimeMs()

        // ── GPU pipeline ──────────────────────────────────────────────────────
        log(.debug, "gpu", "Frame #\(frameCount): running DepthNorm → Dilation → Integrate …")
        processor.process(frame: frame, playerHeads: [])

        let gpuEndMs    = currentTimeMs()
        let depthNormMs = processor.lastDepthNormMs
        let dilationMs  = processor.lastDilationMs
        let integrateMs = processor.lastIntegrateMs
        let gpuTotalMs  = processor.totalComputeMs

        log(.debug, "gpu", String(format:
            "Frame #%d GPU done  norm=%.2fms  dilation=%.2fms  integrate=%.2fms  total=%.2fms",
            frameCount, depthNormMs, dilationMs, integrateMs, gpuTotalMs))

        // ── Marching cubes + send ─────────────────────────────────────────────
        let meshStartMs     = currentTimeMs()
        var bytesSent       = 0
        var chunksGenerated = 0
        var firstSendMs: Int64 = 0

        log(.debug, "mesh", "Frame #\(frameCount): meshing \(allChunks.count) chunks …")

        for chunk in allChunks {
            let rbStart = currentTimeMs()

            let voxData = processor.readbackVolumeRegion(
                start: chunk.voxelStart,
                size:  chunk.voxelSize)

            let rbMs = currentTimeMs() - rbStart

            guard !voxData.isEmpty else {
                log(.debug, "mesh", "Chunk (\(chunk.coord.x),\(chunk.coord.y),\(chunk.coord.z)): readback empty, skip")
                continue
            }

            let meshStart2 = currentTimeMs()
            let mesh = MeshGenerator.generate(
                volume:   voxData,
                voxCount: chunk.voxelSize,
                voxSize:  processor.voxelSize)
            let meshMs2 = currentTimeMs() - meshStart2

            guard !mesh.indices.isEmpty else {
                log(.debug, "mesh", "Chunk (\(chunk.coord.x),\(chunk.coord.y),\(chunk.coord.z)): " +
                    "no surface (readback=\(rbMs)ms)")
                continue
            }

            chunksGenerated += 1
            let sendMs = currentTimeMs()
            if firstSendMs == 0 { firstSendMs = sendMs }

            let pkt = MeshChunkPacket(
                depthTimestampMs:     frame.timestampMs,
                serverSendMs:         sendMs,
                serverComputeStartMs: computeStartMs,
                serverComputeEndMs:   gpuEndMs,
                worldPos:             chunk.worldPos,
                vertices:             mesh.vertices,
                normals:              mesh.normals,
                indices:              mesh.indices,
                serverQueueDepth:     0)

            let payload = serialiseMeshChunk(pkt)
            sendPacket(fd: clientFD, payload: payload)
            let chunkBytes = payload.count + 4
            bytesSent += chunkBytes

            log(.debug, "mesh", String(format:
                "  Chunk (%d,%d,%d)  verts=%d  tris=%d  readback=%lldms  mesh=%lldms  kB=%.1f",
                chunk.coord.x, chunk.coord.y, chunk.coord.z,
                mesh.vertices.count, mesh.indices.count / 3,
                rbMs, meshMs2, Double(chunkBytes) / 1024.0))
        }

        let totalMeshSendMs = Double(currentTimeMs() - meshStartMs)
        let meshOnlyMs      = Double(meshStartMs - gpuEndMs)
        let sendOnlyMs      = totalMeshSendMs - meshOnlyMs
        let roundTripMs     = (firstSendMs > 0 ? firstSendMs : currentTimeMs()) - frame.timestampMs

        totalBytesSent += bytesSent
        totalChunks    += chunksGenerated

        // ── Metrics ───────────────────────────────────────────────────────────
        logger?.log(MetricsLogger.Frame(
            timestampMs:      frame.timestampMs,
            depthNormMs:      depthNormMs,
            dilationMs:       dilationMs,
            integrateMs:      integrateMs,
            meshMs:           meshOnlyMs,
            sendMs:           sendOnlyMs,
            roundTripMs:      roundTripMs,
            bytesSent:        bytesSent,
            chunksGenerated:  chunksGenerated,
            serverQueueDepth: 0))

        // ── Per-10-frame info summary ─────────────────────────────────────────
        if frameCount % 10 == 0 {
            log(.info, "stats", String(format:
                "frame=%d  " +
                "norm=%.1fms  dil=%.1fms  int=%.1fms  " +
                "mesh=%.1fms  send=%.1fms  rtt=%lldms  " +
                "chunks=%d/%d  kB=%.1f  totalKB=%.1f",
                frameCount,
                depthNormMs, dilationMs, integrateMs,
                meshOnlyMs, sendOnlyMs, roundTripMs,
                chunksGenerated, allChunks.count,
                Double(bytesSent) / 1024.0,
                Double(totalBytesSent) / 1024.0))
        }
    }
}

log(.info, "main", "Starting TCP server on port \(port) …")
do {
    try server.start()
} catch {
    log(.error, "main", "Server start failed: \(error)")
    exit(1)
}

log(.info, "main", "Ready — waiting for Quest connection on :\(port)")

// Keep main thread alive (accept loop runs on a background thread)
dispatchMain()
