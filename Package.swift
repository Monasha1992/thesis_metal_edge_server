// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "EdgeMetalServer",
    platforms: [.macOS(.v14)],  // Metal read_write textures, structured concurrency
    targets: [
        .executableTarget(
            name: "EdgeMetalServer",
            path: "Sources/EdgeMetalServer"
        )
    ]
)
