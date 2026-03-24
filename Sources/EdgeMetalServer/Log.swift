import Foundation

// ── Log levels ─────────────────────────────────────────────────────────────────
// Set via --loglevel=debug|info|warn|error   (default: info)
//
// debug  — every frame received, per-chunk mesh output, GPU stage timing
// info   — startup, connection events, every-10-frame summary, metric rows
// warn   — recoverable anomalies (bad packets, empty mesh, parse failures)
// error  — fatal errors written to stderr before exit

enum LogLevel: Int, Comparable {
    case debug = 0, info = 1, warn = 2, error = 3
    static func < (a: LogLevel, b: LogLevel) -> Bool { a.rawValue < b.rawValue }
    var tag: String {
        switch self {
        case .debug: return "DBG"
        case .info:  return "INF"
        case .warn:  return "WRN"
        case .error: return "ERR"
        }
    }
}

/// Active log level — write before any logging starts (set in main.swift).
var currentLogLevel: LogLevel = .info

private let _logLock  = NSLock()
private let _logStart = Date()

/// Emit one log line.  Warnings and errors go to stderr; everything else stdout.
@inline(__always)
func log(_ level: LogLevel = .info, _ subsystem: String, _ msg: String) {
    guard level >= currentLogLevel else { return }
    let t    = String(format: "%8.3f", Date().timeIntervalSince(_logStart))
    let line = "[\(t)s][\(level.tag)][\(subsystem)] \(msg)\n"
    _logLock.lock()
    defer { _logLock.unlock() }
    if level >= .warn {
        fputs(line, stderr)
        fflush(stderr)
    } else {
        print(line, terminator: "")
        fflush(stdout)
    }
}
