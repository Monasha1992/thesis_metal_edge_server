import Foundation

// ─────────────────────────────────────────────────────────────────────────────
// MetricsRecorder.swift — Mac-side per-frame CSV writer
//
// WHAT THIS FILE DOES:
//   Records one CSV row per incoming depth frame on the Mac edge server.
//   Each row captures:
//     - The Mac's wall-clock timestamp (so analysis can correlate to Quest CSVs)
//     - The Quest-side timestamp echoed back in this frame (server_ts_ms on the
//       Quest side = timestamp_echo_ms here — the join key)
//     - Per-stage GPU pipeline timings (parse/upload/dilate/normals/setup/
//       integrate/mesh)
//     - Mesh size produced (vert + triangle counts)
//     - Bytes received (depth frame) and sent (mesh response)
//
//   The Quest CSV is the primary record of the study. This Mac CSV is the
//   server-side detail that explains WHY each Quest `mesh` row has the RTT
//   and mesh size it does. Joined post-hoc in pandas:
//
//     merged = quest_mesh.merge(mac, left_on='server_ts_ms',
//                                     right_on='timestamp_echo_ms',
//                                     how='left')
//
// FILE LIFECYCLE:
//   - One CSV per Mac server SESSION (= one Quest connection).
//     `startSession()` is called from EdgeMetalServer.newConnectionHandler,
//     so each fresh client connection opens a fresh CSV.
//   - Output path: ~/EdgeMetalServer/logs/server_<ISO-timestamp>.csv
//   - Writes are buffered by the OS; we don't flush explicitly. On process
//     kill (Ctrl-C in the terminal running `swift run`), the OS flushes
//     before the file handle is reclaimed.
//
// THREADING:
//   Writes are dispatched onto a dedicated serial queue ("metrics-recorder")
//   so the GPU pipeline thread (.main, where receivePayload runs) never
//   blocks on disk I/O. The queue is `.async` for `record()` and `.sync`
//   for session lifecycle ops (start/end) to keep file-handle state
//   consistent.
// ─────────────────────────────────────────────────────────────────────────────

// @unchecked Sendable: the class has mutable state (fileHandle, currentPath)
// but all access is funnelled through `queue` (a serial DispatchQueue), so
// it is thread-safe by construction. Swift 6's Sendable checker can't see
// through that, hence the `@unchecked` opt-out.
final class MetricsRecorder: @unchecked Sendable {
    static let shared = MetricsRecorder()

    private var fileHandle: FileHandle?
    private var currentPath: URL?
    private let queue = DispatchQueue(label: "metrics-recorder", qos: .utility)

    private init() {}

    // ─────────────────────────────────────────────────────────────────────────
    // startSession — Open a fresh CSV file for the new client connection.
    //
    // Closes any previously-open session first. Creates the logs directory
    // if it doesn't exist. Writes the header row immediately. Subsequent
    // record() calls append rows. The port is baked into the filename so that,
    // when several server instances run at once (one per headset), each one's
    // log is identifiable and two instances started in the same second don't
    // collide.
    // ─────────────────────────────────────────────────────────────────────────
    func startSession(port: UInt16) {
        queue.sync {
            closeUnsafe()

            // Build output path: ~/EdgeMetalServer/logs/server_p<port>_<timestamp>.csv
            let homeDir = FileManager.default.homeDirectoryForCurrentUser
            let logsDir = homeDir.appendingPathComponent("EdgeMetalServer/logs")
            try? FileManager.default.createDirectory(
                at: logsDir, withIntermediateDirectories: true)

            // ISO-ish timestamp safe for filenames (no colons — Finder dislikes them).
            // LOCAL device time for the filename only (human-friendly); the per-row
            // wallMs column below stays UTC epoch-ms so the Quest↔Mac join is unaffected.
            let df = DateFormatter()
            df.dateFormat = "yyyyMMdd_HHmmss"
            df.locale = Locale(identifier: "en_US_POSIX")
            df.timeZone = TimeZone.current
            let stamp = df.string(from: Date())
            let url = logsDir.appendingPathComponent("server_p\(port)_\(stamp).csv")
            currentPath = url

            // Header row — must match the columns documented in
            // docs/06-metrics.md and the Quest-side join expectations.
            let header = """
            wall_ms,timestamp_echo_ms,\
            parse_ms,upload_ms,dilate_ms,normals_ms,setup_ms,integrate_ms,mesh_ms,total_ms,\
            vert_count,tri_count,\
            payload_in_bytes,response_out_bytes

            """

            FileManager.default.createFile(
                atPath: url.path,
                contents: header.data(using: .utf8))

            do {
                fileHandle = try FileHandle(forWritingTo: url)
                try fileHandle?.seekToEnd()
                print("[MetricsRecorder] Session started → \(url.path)")
            } catch {
                print("[MetricsRecorder] Failed to open file: \(error)")
                fileHandle = nil
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // record — Append one row for a just-processed frame.
    //
    // Safe to call when no session is active (early-return). Dispatched
    // asynchronously so the GPU thread isn't blocked on file I/O.
    //
    // All ms values are doubles in milliseconds. payloadInBytes / responseOutBytes
    // include the 5-byte framing header on each direction.
    // ─────────────────────────────────────────────────────────────────────────
    func record(
        timestampEchoMs: UInt64,
        parseMs: Double, uploadMs: Double, dilateMs: Double, normalsMs: Double,
        setupMs: Double, integrateMs: Double, meshMs: Double, totalMs: Double,
        vertCount: Int, triCount: Int,
        payloadInBytes: Int, responseOutBytes: Int
    ) {
        queue.async { [weak self] in
            guard let self = self, let fh = self.fileHandle else { return }

            // Mac wall-clock at the moment the row is composed. Quest's
            // `wall_ms` and this should be within a few ms of each other
            // (both machines have NTP-synced clocks); analysis tolerates
            // ±1 s skew without trouble.
            let wallMs = UInt64(Date().timeIntervalSince1970 * 1000)

            let row = String(
                format: "%llu,%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d\n",
                wallMs, timestampEchoMs,
                parseMs, uploadMs, dilateMs, normalsMs, setupMs, integrateMs, meshMs, totalMs,
                vertCount, triCount,
                payloadInBytes, responseOutBytes
            )

            if let data = row.data(using: .utf8) {
                try? fh.write(contentsOf: data)
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // endSession — Flush and close the active CSV.
    //
    // Currently not called from anywhere in EdgeMetalServer (the OS handles
    // cleanup on process exit). Provided here for tests and for any future
    // explicit-shutdown path.
    // ─────────────────────────────────────────────────────────────────────────
    func endSession() {
        queue.sync { closeUnsafe() }
    }

    private func closeUnsafe() {
        try? fileHandle?.close()
        if let path = currentPath {
            print("[MetricsRecorder] Closed \(path.path)")
        }
        fileHandle = nil
        currentPath = nil
    }
}
