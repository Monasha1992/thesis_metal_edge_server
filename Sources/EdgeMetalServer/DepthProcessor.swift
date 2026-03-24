import Metal
import simd
import Foundation

// ── Uniform layouts (must match struct definitions in Shaders.swift) ──────────

struct DepthUniforms {
    var proj:    (simd_float4x4, simd_float4x4)
    var projInv: (simd_float4x4, simd_float4x4)
    var view:    (simd_float4x4, simd_float4x4)
    var viewInv: (simd_float4x4, simd_float4x4)
    var texSize: SIMD2<Float>
    var zParams: SIMD2<Float>   // x=near, y=far
}

struct EnvUniforms {
    var voxCount:        SIMD3<UInt32>
    var voxSize:         Float
    var voxDist:         Float
    var voxMin:          Float
    var numPlayers:      Int32
    var depthDispThresh: Float
}

struct DilateUniforms {
    var dilateStepSize: Int32
    var texSize:        SIMD2<Float>
    var envVoxDist:     Float
    var envVoxSize:     Float
}

// ── DepthProcessor ────────────────────────────────────────────────────────────
// Owns all Metal GPU resources and runs the TSDF pipeline:
//   DepthNorm → InitDepthDilation → DilateDepthStep×N → Integrate
//
// Volume format:  R32Float  (float per voxel, range [-1, 1], -1 = empty)
// Depth format:   R16Unorm  (matches Quest depth sensor output)
// Normal format:  RGBA32Float

final class DepthProcessor {

    // ── Config ─────────────────────────────────────────────────────────────────
    let voxCount:          SIMD3<Int>
    let voxelSize:         Float
    let voxelDist:         Float    // truncation distance
    let voxelMin:          Float    // minimum surface distance to accept
    let maxUpdateDist:     Float
    let minUpdateDist:     Float
    let depthDilationSteps: Int
    let depthDispThresh:   Float

    // ── Metal objects ──────────────────────────────────────────────────────────
    let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary

    // Pipelines
    private var psClear:          MTLComputePipelineState!
    private var psDepthNorm:      MTLComputePipelineState!
    private var psInitDilation:   MTLComputePipelineState!
    private var psDilateStep:     MTLComputePipelineState!
    private var psIntegrate:      MTLComputePipelineState!

    // Textures
    private(set) var volumeTex:   MTLTexture!   // R32Float, 3D
    private var depthTex:         MTLTexture?   // R16Unorm, 2DArray, 2 slices
    private var normTex:          MTLTexture?   // RGBA32Float, 2DArray, 2 slices
    private var dilationA:        MTLTexture?   // RGBA32Float, 2D
    private var dilationB:        MTLTexture?   // RGBA32Float, 2D

    // Buffers
    private var frustumVolBuf:    MTLBuffer?
    private var playerHeadsBuf:   MTLBuffer!
    private var depthUniBuf:      MTLBuffer!
    private var envUniBuf:        MTLBuffer!
    private var dilateUniBuf:     MTLBuffer!

    // ── Metrics ────────────────────────────────────────────────────────────────
    var lastDepthNormMs:    Double = 0
    var lastDilationMs:     Double = 0
    var lastIntegrateMs:    Double = 0
    var totalComputeMs:     Double = 0

    init(
        device:              MTLDevice,
        voxCount:            SIMD3<Int>  = SIMD3(128, 128, 128),
        voxelSize:           Float       = 0.1,
        voxelDist:           Float       = 0.2,
        voxelMin:            Float       = 0.1,
        maxUpdateDist:       Float       = 6.0,
        minUpdateDist:       Float       = 1.0,
        depthDilationSteps:  Int         = 8,
        depthDispThresh:     Float       = 1.0
    ) throws {
        self.device             = device
        self.voxCount           = voxCount
        self.voxelSize          = voxelSize
        self.voxelDist          = voxelDist
        self.voxelMin           = voxelMin
        self.maxUpdateDist      = maxUpdateDist
        self.minUpdateDist      = minUpdateDist
        self.depthDilationSteps = depthDilationSteps
        self.depthDispThresh    = depthDispThresh

        guard let q = device.makeCommandQueue() else {
            throw ProcessorError.metalInit("makeCommandQueue failed")
        }
        commandQueue = q

        // Compile MSL at runtime — one-time cost on startup
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        library = try device.makeLibrary(source: metalShaderSource, options: options)
        print("[DepthProcessor] MSL compiled on \(device.name)")

        try buildPipelines()
        buildVolume()
        buildFixedBuffers()
    }

    // ── Pipeline setup ─────────────────────────────────────────────────────────

    private func buildPipelines() throws {
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw ProcessorError.metalInit("Function \(name) not found")
            }
            return try device.makeComputePipelineState(function: fn)
        }
        psClear        = try pipeline("kernelClear");        log(.debug, "gpu", "Pipeline: kernelClear compiled")
        psDepthNorm    = try pipeline("kernelDepthNorm");    log(.debug, "gpu", "Pipeline: kernelDepthNorm compiled")
        psInitDilation = try pipeline("kernelInitDepthDilation"); log(.debug, "gpu", "Pipeline: kernelInitDepthDilation compiled")
        psDilateStep   = try pipeline("kernelDilateDepthStep");   log(.debug, "gpu", "Pipeline: kernelDilateDepthStep compiled")
        psIntegrate    = try pipeline("kernelIntegrate");    log(.debug, "gpu", "Pipeline: kernelIntegrate compiled")
        log(.info, "gpu", "All 5 compute pipelines ready")
    }

    private func buildVolume() {
        let desc = MTLTextureDescriptor()
        desc.textureType    = .type3D
        desc.pixelFormat    = .r32Float
        desc.width          = voxCount.x
        desc.height         = voxCount.y
        desc.depth          = voxCount.z
        desc.usage          = [.shaderRead, .shaderWrite]
        desc.storageMode    = .private
        volumeTex = device.makeTexture(descriptor: desc)!
        clearVolume()
    }

    private func buildFixedBuffers() {
        // 512 player head slots (float3 each = 12 bytes)
        playerHeadsBuf = device.makeBuffer(
            length: 512 * MemoryLayout<SIMD3<Float>>.stride,
            options: .storageModeShared)!

        depthUniBuf = device.makeBuffer(
            length: MemoryLayout<DepthUniforms>.size,
            options: .storageModeShared)!

        envUniBuf = device.makeBuffer(
            length: MemoryLayout<EnvUniforms>.size,
            options: .storageModeShared)!

        dilateUniBuf = device.makeBuffer(
            length: MemoryLayout<DilateUniforms>.size,
            options: .storageModeShared)!

        // Write env uniforms once (they don't change per-frame)
        var eu = EnvUniforms(
            voxCount:        SIMD3<UInt32>(UInt32(voxCount.x), UInt32(voxCount.y), UInt32(voxCount.z)),
            voxSize:         voxelSize,
            voxDist:         voxelDist,
            voxMin:          voxelMin,
            numPlayers:      0,
            depthDispThresh: depthDispThresh)
        memcpy(envUniBuf.contents(), &eu, MemoryLayout<EnvUniforms>.size)
    }

    // ── Depth / normal texture management ─────────────────────────────────────

    private func ensureDepthTextures(width: Int, height: Int) {
        if let t = depthTex, t.width == width, t.height == height {
            log(.debug, "gpu", "Depth textures \(width)×\(height) already allocated — reusing")
            return
        }
        log(.info, "gpu", "Allocating depth textures \(width)×\(height)  " +
            "(R16Unorm×2 + RGBA32F×2 norm + RGBA32F×2 dilation ping-pong)")

        let dd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Unorm, width: width, height: height, mipmapped: false)
        dd.textureType  = .type2DArray
        dd.arrayLength  = 2
        dd.usage        = [.shaderRead]
        dd.storageMode  = .shared   // CPU writable for uploads
        depthTex = device.makeTexture(descriptor: dd)!

        let nd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        nd.textureType  = .type2DArray
        nd.arrayLength  = 2
        nd.usage        = [.shaderRead, .shaderWrite]
        nd.storageMode  = .private
        normTex = device.makeTexture(descriptor: nd)!

        // Dilation ping-pong textures
        let diDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        diDesc.usage        = [.shaderRead, .shaderWrite]
        diDesc.storageMode  = .private
        dilationA = device.makeTexture(descriptor: diDesc)!
        dilationB = device.makeTexture(descriptor: diDesc)!
    }

    // ── Upload depth from CPU bytes → Metal texture ────────────────────────────

    private func uploadDepth(pkt: DepthFramePacket) {
        let w = Int(pkt.width), h = Int(pkt.height)
        ensureDepthTextures(width: w, height: h)

        let bytesPerRow = w * 2   // R16Unorm = 2 bytes per pixel
        let region = MTLRegionMake2D(0, 0, w, h)
        let totalBytes = bytesPerRow * h * 2

        let uploadStart = Date()
        pkt.slice0.withUnsafeBytes { ptr in
            depthTex!.replace(region: region, mipmapLevel: 0, slice: 0,
                              withBytes: ptr.baseAddress!, bytesPerRow: bytesPerRow,
                              bytesPerImage: bytesPerRow * h)
        }
        pkt.slice1.withUnsafeBytes { ptr in
            depthTex!.replace(region: region, mipmapLevel: 0, slice: 1,
                              withBytes: ptr.baseAddress!, bytesPerRow: bytesPerRow,
                              bytesPerImage: bytesPerRow * h)
        }
        let uploadMs = Date().timeIntervalSince(uploadStart) * 1000
        log(.debug, "gpu", String(format:
            "Depth uploaded  %d×%d×2 slices  %d bytes  %.2fms",
            w, h, totalBytes, uploadMs))
    }

    // ── Frustum volume (pre-computed per projection matrix) ───────────────────

    private var lastProjForFrustum: simd_float4x4 = .init(0)

    private func rebuildFrustumVolume(proj: simd_float4x4, near: Float) {
        guard !simd_equal(proj, lastProjForFrustum) else { return }
        lastProjForFrustum = proj

        // Extract slopes (view-space tangents) from projection matrix
        // proj.columns[0].x = m00 = 2/(r-l)
        // proj.columns[2].x = m02 = (r+l)/(r-l)
        let m00 = proj.columns.0.x
        let m11 = proj.columns.1.y
        let m02 = proj.columns.2.x
        let m12 = proj.columns.2.y

        let rs = (1.0 + m02) / m00   // right / near
        let ls = (m02 - 1.0) / m00   // left  / near  (negative)
        let ts = (1.0 + m12) / m11   // top   / near
        let bs = (m12 - 1.0) / m11   // bottom/ near  (negative)

        var positions = [SIMD3<Float>]()
        positions.reserveCapacity(200_000)

        var z = near
        while z < maxUpdateDist {
            let xMin = ls * z + voxelSize
            let xMax = rs * z - voxelSize
            let yMin = bs * z + voxelSize
            let yMax = ts * z - voxelSize

            var x = xMin
            while x < xMax {
                var y = yMin
                while y < yMax {
                    let v = SIMD3<Float>(x, y, -z)   // -z: depth sensor forward is -Z
                    let d = simd_length(v)
                    if d > minUpdateDist && d < maxUpdateDist {
                        positions.append(v)
                    }
                    y += voxelSize
                }
                x += voxelSize
            }
            z += voxelSize
        }

        guard !positions.isEmpty else {
            log(.warn, "gpu", "Frustum volume is empty — check near/maxUpdateDist/voxelSize params")
            return
        }

        let byteLen = positions.count * MemoryLayout<SIMD3<Float>>.stride
        frustumVolBuf = device.makeBuffer(bytes: positions,
                                          length: byteLen,
                                          options: .storageModeShared)!
        log(.info, "gpu", String(format:
            "Frustum volume rebuilt  voxels=%d  buffer=%.1f kB  " +
            "slopes(r=%.3f l=%.3f t=%.3f b=%.3f)",
            positions.count, Double(byteLen) / 1024.0,
            rs, ls, ts, bs))
    }

    // ── Main per-frame pipeline ────────────────────────────────────────────────

    func process(frame pkt: DepthFramePacket, playerHeads: [SIMD3<Float>] = []) {
        let t0 = Date()

        // 1. Upload depth pixels
        uploadDepth(pkt: pkt)

        // 2. Rebuild frustum volume if projection changed
        rebuildFrustumVolume(proj: pkt.proj.0, near: pkt.near)
        guard let frustumBuf = frustumVolBuf, frustumBuf.length > 0 else { return }
        let frustumCount = frustumBuf.length / MemoryLayout<SIMD3<Float>>.stride

        // 3. Update depth uniforms
        var du = DepthUniforms(
            proj:    pkt.proj,
            projInv: (simd_inverse(pkt.proj.0), simd_inverse(pkt.proj.1)),
            view:    pkt.view,
            viewInv: (simd_inverse(pkt.view.0), simd_inverse(pkt.view.1)),
            texSize: SIMD2<Float>(Float(pkt.width), Float(pkt.height)),
            zParams: SIMD2<Float>(pkt.near, pkt.far))
        memcpy(depthUniBuf.contents(), &du, MemoryLayout<DepthUniforms>.size)

        // 4. Update player heads buffer
        var eu = envUniBuf.contents().load(as: EnvUniforms.self)
        eu.numPlayers = Int32(min(playerHeads.count, 512))
        memcpy(envUniBuf.contents(), &eu, MemoryLayout<EnvUniforms>.size)
        if !playerHeads.isEmpty {
            let stride = MemoryLayout<SIMD3<Float>>.stride
            let count  = min(playerHeads.count, 512)
            memcpy(playerHeadsBuf.contents(), playerHeads, count * stride)
        }

        guard let cmd = commandQueue.makeCommandBuffer() else {
            log(.error, "gpu", "makeCommandBuffer() returned nil — Metal device lost?")
            return
        }
        log(.debug, "gpu", "Encoding GPU pipeline  frustumVoxels=\(frustumCount)  " +
            "dilationSteps=\(depthDilationSteps)")

        // ── Stage A: DepthNorm ────────────────────────────────────────────────
        let tA = Date()
        encodeDepthNorm(cmd: cmd, du: du)
        log(.debug, "gpu", "  [A] DepthNorm encoded")
        let tB = Date()

        // ── Stage B: Depth dilation ───────────────────────────────────────────
        encodeDepthDilation(cmd: cmd, du: du)
        log(.debug, "gpu", "  [B] DepthDilation encoded  steps=\(depthDilationSteps)")
        let tC = Date()

        // ── Stage C: TSDF Integrate ───────────────────────────────────────────
        encodeIntegrate(cmd: cmd, frustumBuf: frustumBuf, frustumCount: frustumCount)
        log(.debug, "gpu", "  [C] Integrate encoded  frustumVoxels=\(frustumCount)")

        cmd.addCompletedHandler { [weak self] _ in
            let tD = Date()
            guard let self else { return }
            self.lastDepthNormMs  = tB.timeIntervalSince(tA) * 1000
            self.lastDilationMs   = tC.timeIntervalSince(tB) * 1000
            self.lastIntegrateMs  = tD.timeIntervalSince(tC) * 1000
            self.totalComputeMs   = tD.timeIntervalSince(t0) * 1000
            log(.debug, "gpu", String(format:
                "GPU done  norm=%.2fms  dilation=%.2fms  integrate=%.2fms  total=%.2fms",
                self.lastDepthNormMs, self.lastDilationMs,
                self.lastIntegrateMs, self.totalComputeMs))
        }

        cmd.commit()
        log(.debug, "gpu", "Command buffer committed — waiting for GPU …")
        cmd.waitUntilCompleted()   // block until GPU is done before meshing
        log(.debug, "gpu", "GPU completed")
    }

    // ── Encode helpers ────────────────────────────────────────────────────────

    private func encodeDepthNorm(cmd: MTLCommandBuffer, du: DepthUniforms) {
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psDepthNorm)
        enc.setTexture(depthTex,  index: 0)
        enc.setTexture(normTex,   index: 1)
        enc.setBuffer(depthUniBuf, offset: 0, index: 0)
        let w = depthTex!.width, h = depthTex!.height
        // 2 slices (stereo)
        let tg = MTLSize(width: 8, height: 8, depth: 1)
        let grid = MTLSize(width: (w + 7) / 8, height: (h + 7) / 8, depth: 2)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    private func encodeDepthDilation(cmd: MTLCommandBuffer, du: DepthUniforms) {
        let w = depthTex!.width, h = depthTex!.height

        // Init: copy depth slice 0 into dilationA
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(psInitDilation)
            enc.setTexture(depthTex,  index: 0)
            enc.setTexture(dilationA, index: 1)
            let tg   = MTLSize(width: 8, height: 8, depth: 1)
            let grid = MTLSize(width: (w + 7) / 8, height: (h + 7) / 8, depth: 1)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            enc.endEncoding()
        }

        // Dilation steps (ping-pong A↔B)
        var stepSize = 1 << depthDilationSteps
        var src = dilationA!, dst = dilationB!

        for _ in 0..<depthDilationSteps {
            var du2 = DilateUniforms(
                dilateStepSize: Int32(stepSize),
                texSize: SIMD2<Float>(Float(w), Float(h)),
                envVoxDist: voxelDist,
                envVoxSize: voxelSize)
            memcpy(dilateUniBuf.contents(), &du2, MemoryLayout<DilateUniforms>.size)
            stepSize /= 2

            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(psDilateStep)
                enc.setTexture(src, index: 0)
                enc.setTexture(dst, index: 1)
                enc.setBuffer(dilateUniBuf, offset: 0, index: 0)
                let tg   = MTLSize(width: 8, height: 8, depth: 1)
                let grid = MTLSize(width: (w + 7) / 8, height: (h + 7) / 8, depth: 1)
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
                enc.endEncoding()
            }
            swap(&src, &dst)
        }
        // After ping-pong, src holds the final dilated depth — reassign dilationA
        dilationA = src
    }

    private func encodeIntegrate(cmd: MTLCommandBuffer,
                                  frustumBuf: MTLBuffer, frustumCount: Int) {
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psIntegrate)
        enc.setTexture(volumeTex,  index: 0)
        enc.setTexture(depthTex,   index: 1)
        enc.setTexture(normTex,    index: 2)
        enc.setTexture(dilationA,  index: 3)
        enc.setBuffer(frustumBuf,     offset: 0, index: 0)
        enc.setBuffer(depthUniBuf,    offset: 0, index: 1)
        enc.setBuffer(envUniBuf,      offset: 0, index: 2)
        enc.setBuffer(playerHeadsBuf, offset: 0, index: 3)

        let tg   = MTLSize(width: 64, height: 1, depth: 1)
        let grid = MTLSize(width: (frustumCount + 63) / 64, height: 1, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    // ── Volume readback (CPU, for marching cubes) ──────────────────────────────
    // Returns the TSDF voxels for a world-space region as a flat Float array.
    // start/size are in voxel coordinates.

    func readbackVolumeRegion(start: SIMD3<Int>, size: SIMD3<Int>) -> [Float] {
        let count = size.x * size.y * size.z
        let byteLen = count * MemoryLayout<Float>.size

        guard let stagingBuf = device.makeBuffer(length: byteLen, options: .storageModeShared) else {
            log(.warn, "gpu", "readbackVolumeRegion: makeBuffer(\(byteLen)) failed  start=\(start)  size=\(size)")
            return []
        }

        guard let cmd = commandQueue.makeCommandBuffer(),
              let blit = cmd.makeBlitCommandEncoder() else {
            log(.warn, "gpu", "readbackVolumeRegion: failed to create command buffer or blit encoder")
            return []
        }
        log(.debug, "gpu", "Readback  start=(\(start.x),\(start.y),\(start.z))  " +
            "size=(\(size.x),\(size.y),\(size.z))  kB=\(byteLen/1024)")

        blit.copy(
            from:              volumeTex,
            sourceSlice:       0,
            sourceLevel:       0,
            sourceOrigin:      MTLOrigin(x: start.x, y: start.y, z: start.z),
            sourceSize:        MTLSize(width: size.x, height: size.y, depth: size.z),
            to:                stagingBuf,
            destinationOffset: 0,
            destinationBytesPerRow:   size.x * MemoryLayout<Float>.size,
            destinationBytesPerImage: size.x * size.y * MemoryLayout<Float>.size)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let ptr = stagingBuf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    // ── Clear volume ──────────────────────────────────────────────────────────

    func clearVolume() {
        guard let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psClear)
        enc.setTexture(volumeTex, index: 0)
        let tg = MTLSize(width: 4, height: 4, depth: 4)
        let grid = MTLSize(
            width:  (voxCount.x + 3) / 4,
            height: (voxCount.y + 3) / 4,
            depth:  (voxCount.z + 3) / 4)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        log(.info, "gpu", "Volume cleared  \(voxCount.x)×\(voxCount.y)×\(voxCount.z) voxels  " +
            "(\(voxCount.x*voxCount.y*voxCount.z) total)")
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    func worldToVoxel(_ pos: SIMD3<Float>) -> SIMD3<Int> {
        var p = pos / voxelSize
        p += SIMD3<Float>(Float(voxCount.x), Float(voxCount.y), Float(voxCount.z)) / 2.0
        return SIMD3<Int>(
            max(0, min(Int(p.x), voxCount.x - 1)),
            max(0, min(Int(p.y), voxCount.y - 1)),
            max(0, min(Int(p.z), voxCount.z - 1)))
    }

    enum ProcessorError: Error {
        case metalInit(String)
    }
}
