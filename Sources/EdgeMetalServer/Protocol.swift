import Foundation
import simd

// ── Wire protocol ─────────────────────────────────────────────────────────────
// Must stay in sync with Assets/Anaglyph/EdgeServer/EdgeProtocol.cs
//
// Framing:  [4 bytes LE int32 payload length] [N bytes payload]
// Payload:  first byte = PacketType, then packet-specific fields.
//
// All floats little-endian. Matrices written row-major from C# (row 0 col 0..3,
// row 1 col 0..3, …) and must be transposed when loading into column-major
// simd_float4x4.

enum PacketType: UInt8 {
    case sync       = 0x00
    case depthFrame = 0x01
    case meshChunk  = 0x02
}

// ── DepthFrame packet (Quest → Mac) ──────────────────────────────────────────
struct DepthFramePacket {
    var timestampMs: Int64       // Unix ms, used for latency measurement
    var width:  Int32
    var height: Int32
    var proj:   (simd_float4x4, simd_float4x4)   // stereo projection matrices
    var view:   (simd_float4x4, simd_float4x4)   // stereo view matrices
    var near:   Float
    var far:    Float
    var slice0: Data             // R16Unorm raw bytes, left eye
    var slice1: Data             // R16Unorm raw bytes, right eye
}

// ── MeshChunk packet (Mac → Quest) ────────────────────────────────────────────
struct MeshChunkPacket {
    var depthTimestampMs:     Int64    // which depth frame produced this mesh
    var serverSendMs:         Int64
    var serverComputeStartMs: Int64
    var serverComputeEndMs:   Int64
    var worldPos:             SIMD3<Float>
    var vertices:             [SIMD3<Float>]
    var normals:              [SIMD3<Float>]
    var indices:              [Int32]
    var serverQueueDepth:     Int32
}

// ── Framed I/O ─────────────────────────────────────────────────────────────────

func readExact(fd: Int32, count: Int) -> Data? {
    var data = Data(count: count)
    var received = 0
    while received < count {
        let n = data.withUnsafeMutableBytes { ptr in
            Darwin.recv(fd, ptr.baseAddress! + received, count - received, 0)
        }
        if n <= 0 { return nil }
        received += n
    }
    return data
}

func readPacket(fd: Int32) -> Data? {
    guard let lenData = readExact(fd: fd, count: 4) else {
        log(.debug, "proto", "readExact(4) returned nil — connection closed")
        return nil
    }
    let length = lenData.withUnsafeBytes { $0.load(as: Int32.self) }
    guard length > 0, length < 64 * 1024 * 1024 else {
        log(.warn, "proto", "Invalid packet length \(length) — expected 1…\(64*1024*1024-1), dropping connection")
        return nil
    }
    log(.debug, "proto", "Reading packet body  length=\(length) bytes")
    guard let body = readExact(fd: fd, count: Int(length)) else {
        log(.warn, "proto", "readExact(\(length)) returned nil mid-packet — connection lost")
        return nil
    }
    return body
}

func sendPacket(fd: Int32, payload: Data) {
    var length = Int32(payload.count)
    let hdrSent = withUnsafeBytes(of: &length) { Darwin.send(fd, $0.baseAddress!, 4, 0) }
    if hdrSent != 4 {
        log(.warn, "proto", "sendPacket header: sent \(hdrSent)/4 bytes  errno=\(errno)")
    }
    let bodySent = payload.withUnsafeBytes { Darwin.send(fd, $0.baseAddress!, payload.count, 0) }
    if bodySent != payload.count {
        log(.warn, "proto", "sendPacket body: sent \(bodySent)/\(payload.count) bytes  errno=\(errno)")
    }
    log(.debug, "proto", "Sent packet  headerOK=\(hdrSent==4)  body=\(bodySent)/\(payload.count) bytes")
}

// ── Deserialise ────────────────────────────────────────────────────────────────

func parseDepthFrame(_ data: Data) -> DepthFramePacket? {
    var offset = 1   // skip PacketType byte
    guard data.count > offset else { return nil }

    func readInt64() -> Int64 {
        let v = data[offset..<offset+8].withUnsafeBytes { $0.load(as: Int64.self) }; offset += 8; return v
    }
    func readInt32() -> Int32 {
        let v = data[offset..<offset+4].withUnsafeBytes { $0.load(as: Int32.self) }; offset += 4; return v
    }
    func readFloat() -> Float {
        let v = data[offset..<offset+4].withUnsafeBytes { $0.load(as: Float.self) }; offset += 4; return v
    }
    func readMatrix() -> simd_float4x4 {
        // C# writes row-major; convert to column-major for simd_float4x4
        var flat = [Float](repeating: 0, count: 16)
        for i in 0..<16 { flat[i] = readFloat() }
        return simd_float4x4(columns: (
            SIMD4<Float>(flat[0], flat[4], flat[8],  flat[12]),
            SIMD4<Float>(flat[1], flat[5], flat[9],  flat[13]),
            SIMD4<Float>(flat[2], flat[6], flat[10], flat[14]),
            SIMD4<Float>(flat[3], flat[7], flat[11], flat[15])
        ))
    }

    let ts  = readInt64()
    let w   = readInt32()
    let h   = readInt32()
    let p0  = readMatrix()
    let p1  = readMatrix()
    let v0  = readMatrix()
    let v1  = readMatrix()
    let nz  = readFloat()
    let fz  = readFloat()

    let sliceBytes = Int(w) * Int(h) * 2   // R16 = 2 bytes/pixel
    guard offset + sliceBytes * 2 <= data.count else { return nil }

    let s0 = data[offset..<offset+sliceBytes]; offset += sliceBytes
    let s1 = data[offset..<offset+sliceBytes]

    let pkt = DepthFramePacket(
        timestampMs: ts, width: w, height: h,
        proj: (p0, p1), view: (v0, v1),
        near: nz, far: fz,
        slice0: Data(s0), slice1: Data(s1))

    log(.debug, "proto", String(format:
        "Parsed DepthFrame  ts=%lldms  size=%d×%d  near=%.3f  far=%.3f  " +
        "sliceBytes=%d×2  totalPayload=%d",
        ts, w, h, nz, fz, sliceBytes, data.count))

    return pkt
}

// ── Serialise ──────────────────────────────────────────────────────────────────

func serialiseMeshChunk(_ pkt: MeshChunkPacket) -> Data {
    var out = Data()

    func appendInt8 (_ v: Int8)   { out.append(UInt8(bitPattern: v)) }
    func appendUInt8 (_ v: UInt8) { out.append(v) }
    func appendInt32 (_ v: Int32)  { withUnsafeBytes(of: v) { out.append(contentsOf: $0) } }
    func appendInt64 (_ v: Int64)  { withUnsafeBytes(of: v) { out.append(contentsOf: $0) } }
    func appendFloat (_ v: Float)  { withUnsafeBytes(of: v) { out.append(contentsOf: $0) } }
    func appendVec3  (_ v: SIMD3<Float>) { appendFloat(v.x); appendFloat(v.y); appendFloat(v.z) }

    appendUInt8(PacketType.meshChunk.rawValue)
    appendInt64(pkt.depthTimestampMs)
    appendInt64(pkt.serverSendMs)
    appendInt64(pkt.serverComputeStartMs)
    appendInt64(pkt.serverComputeEndMs)
    appendVec3(pkt.worldPos)
    appendInt32(Int32(pkt.vertices.count))
    appendInt32(Int32(pkt.indices.count))
    pkt.vertices.forEach { appendVec3($0) }
    pkt.normals.forEach  { appendVec3($0) }
    pkt.indices.forEach  { appendInt32($0) }
    appendInt32(pkt.serverQueueDepth)

    return out
}
