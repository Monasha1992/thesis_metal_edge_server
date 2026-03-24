import Foundation

// ── Per-frame performance metrics CSV logger ──────────────────────────────────
// Creates (or overwrites) a CSV at the given path, writes a header row, then
// appends one row per processed depth frame.  Thread-safe via NSLock.
//
// CSV columns:
//   timestampMs        — Unix ms of the depth frame (from Quest)
//   depthNormMs        — GPU time for kernelDepthNorm
//   dilationMs         — GPU time for depth dilation pass
//   integrateMs        — GPU time for kernelIntegrate (TSDF)
//   meshMs             — CPU time for all marching-cubes passes
//   sendMs             — TCP send time for all chunks
//   roundTripMs        — frame.timestampMs → server sends first chunk
//   bytesSent          — total bytes sent for this frame (including framing)
//   chunksGenerated    — number of non-empty mesh chunks sent
//   serverQueueDepth   — depth of incoming frame queue at dispatch time

final class MetricsLogger {

    struct Frame {
        var timestampMs:      Int64
        var depthNormMs:      Double
        var dilationMs:       Double
        var integrateMs:      Double
        var meshMs:           Double
        var sendMs:           Double
        var roundTripMs:      Int64
        var bytesSent:        Int
        var chunksGenerated:  Int
        var serverQueueDepth: Int32
    }

    private let fileHandle: FileHandle
    private let lock = NSLock()

    init(path: String) throws {
        FileManager.default.createFile(atPath: path, contents: nil, attributes: nil)
        fileHandle = try FileHandle(forWritingTo: URL(fileURLWithPath: path))

        let header = "timestampMs,depthNormMs,dilationMs,integrateMs," +
                     "meshMs,sendMs,roundTripMs,bytesSent,chunksGenerated,serverQueueDepth\n"
        fileHandle.write(Data(header.utf8))
    }

    func log(_ m: Frame) {
        let row = String(format: "%lld,%.3f,%.3f,%.3f,%.3f,%.3f,%lld,%d,%d,%d\n",
                         m.timestampMs,
                         m.depthNormMs, m.dilationMs, m.integrateMs,
                         m.meshMs, m.sendMs,
                         m.roundTripMs, m.bytesSent,
                         m.chunksGenerated, m.serverQueueDepth)
        lock.lock()
        fileHandle.write(Data(row.utf8))
        lock.unlock()
    }

    deinit {
        try? fileHandle.close()
    }
}
