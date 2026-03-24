import Foundation
import Darwin

// ── POSIX TCP server ──────────────────────────────────────────────────────────
// Accepts one Quest client at a time on the specified port.
// The client fd is passed to the provided callback which runs in a background thread.
// The server continues listening for a new connection after each disconnect.

final class TCPServer {
    let port: UInt16
    private var serverFD: Int32 = -1

    // Called from a background thread with the connected client fd.
    // The callback is responsible for closing the fd when done.
    var onClientConnected: ((Int32) -> Void)?

    init(port: UInt16) {
        self.port = port
    }

    func start() throws {
        serverFD = socket(AF_INET, SOCK_STREAM, 0)
        guard serverFD >= 0 else {
            log(.error, "tcp", "socket() failed  errno=\(errno)")
            throw ServerError.socketFailed
        }
        log(.debug, "tcp", "Socket created  fd=\(serverFD)")

        // SO_REUSEADDR so we can restart quickly
        var yes: Int32 = 1
        let reuseRet = setsockopt(serverFD, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        log(.debug, "tcp", "SO_REUSEADDR \(reuseRet == 0 ? "set" : "FAILED errno=\(errno)")")

        // TCP_NODELAY — disable Nagle (critical for latency measurement)
        let noDelayRet = setsockopt(serverFD, IPPROTO_TCP, TCP_NODELAY, &yes, socklen_t(MemoryLayout<Int32>.size))
        log(.debug, "tcp", "TCP_NODELAY \(noDelayRet == 0 ? "set" : "FAILED errno=\(errno)")")

        var addr = sockaddr_in()
        addr.sin_family      = sa_family_t(AF_INET)
        addr.sin_port        = port.bigEndian
        addr.sin_addr.s_addr = INADDR_ANY

        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(serverFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            log(.error, "tcp", "bind() failed on port \(port)  errno=\(errno)")
            throw ServerError.bindFailed
        }
        log(.debug, "tcp", "Bound to 0.0.0.0:\(port)")

        guard listen(serverFD, 1) == 0 else {
            log(.error, "tcp", "listen() failed  errno=\(errno)")
            throw ServerError.listenFailed
        }
        log(.info, "tcp", "Listening on 0.0.0.0:\(port)  backlog=1")

        // Accept loop runs in a background thread so main can remain free
        Thread.detachNewThread { [weak self] in
            self?.acceptLoop()
        }
    }

    private func acceptLoop() {
        while true {
            var clientAddr = sockaddr_in()
            var addrLen = socklen_t(MemoryLayout<sockaddr_in>.size)
            let clientFD = withUnsafeMutablePointer(to: &clientAddr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    accept(serverFD, $0, &addrLen)
                }
            }
            guard clientFD >= 0 else {
                log(.warn, "tcp", "accept() returned \(clientFD)  errno=\(errno)  retrying …")
                continue
            }

            // TCP_NODELAY on client socket too
            var yes: Int32 = 1
            let ndRet = setsockopt(clientFD, IPPROTO_TCP, TCP_NODELAY, &yes, socklen_t(MemoryLayout<Int32>.size))
            log(.debug, "tcp", "Client TCP_NODELAY \(ndRet == 0 ? "set" : "FAILED")")

            // Resolve peer address
            var hostBuf = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            var portBuf = [CChar](repeating: 0, count: Int(NI_MAXSERV))
            _ = withUnsafePointer(to: clientAddr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    getnameinfo($0, addrLen,
                                &hostBuf, socklen_t(hostBuf.count),
                                &portBuf, socklen_t(portBuf.count),
                                NI_NUMERICHOST | NI_NUMERICSERV)
                }
            }
            let peerIP   = String(cString: hostBuf)
            let peerPort = String(cString: portBuf)
            log(.info, "tcp", "Quest connected  fd=\(clientFD)  peer=\(peerIP):\(peerPort)")

            // Handle client in its own thread
            Thread.detachNewThread { [weak self] in
                self?.onClientConnected?(clientFD)
                close(clientFD)
                log(.info, "tcp", "Client fd=\(clientFD) socket closed  peer=\(peerIP):\(peerPort)")
            }
        }
    }

    func stop() {
        if serverFD >= 0 {
            log(.info, "tcp", "Stopping server  fd=\(serverFD)")
            close(serverFD)
            serverFD = -1
        }
    }

    enum ServerError: Error {
        case socketFailed, bindFailed, listenFailed
    }
}
