// swift-tools-version: 6.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "EdgeMetalServer",
    platforms: [.macOS(.v14)],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .executableTarget(
            name: "EdgeMetalServer",
            // Declaring the Shaders directory as a processed resource makes
            // SwiftPM compile the .metal files into default.metallib inside
            // the target's resource bundle AND synthesize `Bundle.module`,
            // which MetalPipeline.swift needs for makeDefaultLibrary(bundle:).
            // Without this, `swift build` fails with "Bundle has no member
            // 'module'" (Xcode-based builds resolved it differently).
            resources: [
                .process("Shaders/DepthDilation.metal"),
                .process("Shaders/DepthNormal.metal"),
                .process("Shaders/DepthProcess.metal"),
                .process("Shaders/SurfaceNets.metal"),
                .process("Shaders/VolumeIntegration.metal")
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
            ]
        ),
        .testTarget(
            name: "EdgeMetalServerTests",
            dependencies: ["EdgeMetalServer"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
