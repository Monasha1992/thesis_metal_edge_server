import Foundation
import Network

@main
struct EdgeMetalServer {
    static let metal = MetalPipeline()

    static func main() {
        let port: UInt16 = 9876

        let listener = try! NWListener(using: .tcp, on: NWEndpoint.Port(rawValue: port)!)

        listener.newConnectionHandler = { connection in
            print("Client connected: \(connection.endpoint)")

            connection.start(queue: .main)
            Self.receive(on: connection)
        }

        listener.start(queue: .main)
        print("Server listening on port \(port)")

        dispatchMain()
    }

    static func receive(on connection: NWConnection) {
        // Read 5-byte header
        connection.receive(minimumIncompleteLength: 5, maximumLength: 5) { data, _, _, error in
            if let error = error {
                print("Error: \(error)")
                return
            }

            guard let data = data, data.count == 5 else {
                print("Connection closed")
                return
            }

            let type = data[0]
            let length = Int(data[1]) << 24 | Int(data[2]) << 16 | Int(data[3]) << 8 | Int(data[4])

            // Read exact payload
            receivePayload(on: connection, type: type, remaining: length, accumulated: Data())
        }
    }

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
                // Need more bytes
                receivePayload(
                    on: connection, type: type, remaining: left, accumulated: accumulated)
            } else {
                // Full message received
                let frame = parseDepthFrame(accumulated)
                let depthTexture = metal.uploadDepthTexture(frame: frame)
                let dilatedDepth = metal.dilateDepth(depthTexture: depthTexture, frame: frame)
                let normalsTexture = metal.generateNormals(depthTexture: depthTexture, frame: frame)
                metal.setupVolume(frame: frame)
                metal.setupFrustum(frame: frame)
                metal.integrateDepth(
                    depthTexture: depthTexture,
                    normTexture: normalsTexture,
                    dilatedDepth: dilatedDepth,
                    frame: frame
                )
                
                // Extract and send volume data
                let voxels = metal.extractNonEmptyVoxels()
                let voxelData = MetalPipeline.serializeVoxels(voxels)

                // Send with header: type 0x02, then length, then payload
                var response = Data()
                response.append(0x02)  // message type: volume data
                var len = UInt32(voxelData.count)
                let lenBytes = Data([
                    UInt8((len >> 24) & 0xFF),
                    UInt8((len >> 16) & 0xFF),
                    UInt8((len >> 8) & 0xFF),
                    UInt8(len & 0xFF)
                ])
                response.append(lenBytes)
                response.append(voxelData)

                connection.send(content: response, completion: .contentProcessed { _ in
                    print("Sent \(voxels.count) voxels (\(response.count) bytes)")
                })

                // Listen for next message
                receive(on: connection)

            }
        }
    }
}
