import Foundation
import Metal
import simd

class MetalPipeline: @unchecked Sendable {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let depthStatsPipeline: MTLComputePipelineState
    let initDilationPipeline: MTLComputePipelineState
    let dilatePipeline: MTLComputePipelineState
    
    var dilationA: MTLTexture?
    var dilationB: MTLTexture?
    
    var volumeTexture: MTLTexture?
    var frustumBuffer: MTLBuffer?
    var frustumPointCount: Int = 0
    
    let clearVolumePipeline: MTLComputePipelineState
    let integratePipeline: MTLComputePipelineState
    
    let normalsPipeline: MTLComputePipelineState
    var normalsTexture: MTLTexture?
    
    let extractPipeline: MTLComputePipelineState

    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device found")
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        let library = try! device.makeDefaultLibrary(bundle: Bundle.module)
        let statsFunc = library.makeFunction(name: "depthStats")!
        self.depthStatsPipeline = try! device.makeComputePipelineState(function: statsFunc)
        
        let initDilationFunc = library.makeFunction(name: "initDepthDilation")!
        self.initDilationPipeline = try! device.makeComputePipelineState(function: initDilationFunc)

        let dilateFunc = library.makeFunction(name: "dilateDepthStep")!
        self.dilatePipeline = try! device.makeComputePipelineState(function: dilateFunc)

        let clearFunc = library.makeFunction(name: "clearVolume")!
        self.clearVolumePipeline = try! device.makeComputePipelineState(function: clearFunc)

        let integrateFunc = library.makeFunction(name: "integrate")!
        self.integratePipeline = try! device.makeComputePipelineState(function: integrateFunc)

        let normalsFunc = library.makeFunction(name: "depthNormals")!
        self.normalsPipeline = try! device.makeComputePipelineState(function: normalsFunc)
        
        let extractFunc = library.makeFunction(name: "extractNonEmpty")!
        self.extractPipeline = try! device.makeComputePipelineState(function: extractFunc)

        print("Metal device: \(device.name)")
        print("IntegrateParams size: \(MemoryLayout<IntegrateParams>.size), stride: \(MemoryLayout<IntegrateParams>.stride)")

    }

    func uploadDepthTexture(frame: DepthFrame) -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: frame.width,
            height: frame.height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]

        let texture = device.makeTexture(descriptor: descriptor)!

        frame.depthPixels.withUnsafeBytes { ptr in
            texture.replace(
                region: MTLRegionMake2D(0, 0, frame.width, frame.height),
                mipmapLevel: 0,
                withBytes: ptr.baseAddress!,
                bytesPerRow: frame.width * 4
            )
        }

        return texture
    }
    
    func computeDepthStats(texture: MTLTexture) -> (min: Float, max: Float) {
        let minBuffer = device.makeBuffer(length: 4, options: .storageModeShared)!
        let maxBuffer = device.makeBuffer(length: 4, options: .storageModeShared)!

        // Init min to large value, max to 0
        minBuffer.contents().storeBytes(of: UInt32(0xFFFFFFFF), as: UInt32.self)
        maxBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(depthStatsPipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(minBuffer, offset: 0, index: 0)
        encoder.setBuffer(maxBuffer, offset: 0, index: 1)

        let w = depthStatsPipeline.threadExecutionWidth
        let h = depthStatsPipeline.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: texture.width, height: texture.height, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let minInt = minBuffer.contents().load(as: UInt32.self)
        let maxInt = maxBuffer.contents().load(as: UInt32.self)

        return (Float(minInt) / 10000.0, Float(maxInt) / 10000.0)
    }

    struct DilationParams {
        var proj: float4x4          // 64 bytes
        var texSize: SIMD2<UInt32>  // 8 bytes
        var voxDist: Float          // 4 bytes
        var voxSize: Float          // 4 bytes
        var stepSize: Int32         // 4 bytes
        var _pad1: Int32 = 0        // 4 bytes
        var _pad2: Int64 = 0        // 8 bytes → total 96
    }

    func createDilationTexture(width: Int, height: Int) -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)!
    }

    func dilateDepth(depthTexture: MTLTexture, frame: DepthFrame,
                     voxelSize: Float = 0.1, voxelDist: Float = 0.2,
                     dilationSteps: Int = 8) -> MTLTexture {
        let w = depthTexture.width
        let h = depthTexture.height

        // Create ping-pong textures if needed
        if dilationA == nil || dilationA!.width != w || dilationA!.height != h {
            dilationA = createDilationTexture(width: w, height: h)
            dilationB = createDilationTexture(width: w, height: h)
        }

        let commandBuffer = commandQueue.makeCommandBuffer()!

        // Init
        let initEncoder = commandBuffer.makeComputeCommandEncoder()!
        initEncoder.setComputePipelineState(initDilationPipeline)
        initEncoder.setTexture(depthTexture, index: 0)
        initEncoder.setTexture(dilationA!, index: 1)
        let threads = MTLSize(width: w, height: h, depth: 1)
        let groupW = initDilationPipeline.threadExecutionWidth
        let groupH = initDilationPipeline.maxTotalThreadsPerThreadgroup / groupW
        let threadsPerGroup = MTLSize(width: groupW, height: groupH, depth: 1)
        initEncoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerGroup)
        initEncoder.endEncoding()

        // Dilate steps
        var maxStep = 1
        for _ in 0..<dilationSteps { maxStep *= 2 }

        var stepSize = maxStep
        var texA = dilationA!
        var texB = dilationB!

        for _ in 0..<maxStep {
            var params = DilationParams(
                proj: frame.proj[0],
                texSize: SIMD2<UInt32>(UInt32(w), UInt32(h)),
                voxDist: voxelDist,
                voxSize: voxelSize,
                stepSize: Int32(stepSize)
            )

            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(dilatePipeline)
            encoder.setTexture(texA, index: 0)
            encoder.setTexture(texB, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<DilationParams>.size, index: 0)

            let dGroupW = dilatePipeline.threadExecutionWidth
            let dGroupH = dilatePipeline.maxTotalThreadsPerThreadgroup / dGroupW
            encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: dGroupW, height: dGroupH, depth: 1))
            encoder.endEncoding()

            stepSize /= 2
            swap(&texA, &texB)
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return texA
    }
    
    func setupVolume(frame: DepthFrame) {
        if volumeTexture != nil { return } // already set up

        let desc = MTLTextureDescriptor()
        desc.textureType = .type3D
        desc.pixelFormat = .r32Float
        desc.width = Int(frame.voxelCount.x)
        desc.height = Int(frame.voxelCount.y)
        desc.depth = Int(frame.voxelCount.z)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .managed

        volumeTexture = device.makeTexture(descriptor: desc)!
        print("Created volume: \(volumeTexture!.width)x\(volumeTexture!.height)x\(volumeTexture!.depth)")
        
        clearVolume()
        print("Volume cleared")
    }

    // same logic as EnvironmentMapper.Setup() in C#
    func setupFrustum(frame: DepthFrame) {
        if frustumBuffer != nil { return } // already set up

        let proj = frame.proj[0]
        let voxelSize = frame.voxelSize
        let maxDist = frame.maxUpdateDist
        let minDist: Float = 1.0

        // Decompose projection to get frustum slopes
        let near = abs(proj[2][3] * 0.5)
        let ls = proj[2][0] / proj[0][0]  // left slope
        let rs = (2.0 - proj[2][0] * proj[0][0]) / proj[0][0] // approximation

        // Simpler: use inverse projection to get corner rays
        let projInv = frame.projInv[0]

        // Get frustum corners at z=1
        func unprojectCorner(ndcX: Float, ndcY: Float) -> SIMD3<Float> {
            let clip = SIMD4<Float>(ndcX, ndcY, -1, 1)
            let view = projInv * clip
            return SIMD3<Float>(view.x, view.y, view.z) / view.w
        }

        let bl = unprojectCorner(ndcX: -1, ndcY: -1)
        let br = unprojectCorner(ndcX:  1, ndcY: -1)
        let tl = unprojectCorner(ndcX: -1, ndcY:  1)
        let tr = unprojectCorner(ndcX:  1, ndcY:  1)

        // Slopes (direction per unit Z)
        let leftSlope = bl.x / (-bl.z)
        let rightSlope = br.x / (-br.z)
        let bottomSlope = bl.y / (-bl.z)
        let topSlope = tl.y / (-tl.z)

        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(200000)

        var z: Float = near
        while z < maxDist {
            let xMin = leftSlope * z + voxelSize
            let xMax = rightSlope * z - voxelSize
            let yMin = bottomSlope * z + voxelSize
            let yMax = topSlope * z - voxelSize

            var x = xMin
            while x < xMax {
                var y = yMin
                while y < yMax {
                    let v = SIMD3<Float>(x, y, -z)
                    let mag = length(v)
                    if mag > minDist && mag < maxDist {
                        positions.append(v)
                    }
                    y += voxelSize
                }
                x += voxelSize
            }
            z += voxelSize
        }

        frustumPointCount = positions.count
        frustumBuffer = device.makeBuffer(
            bytes: positions,
            length: positions.count * MemoryLayout<SIMD3<Float>>.stride,
            options: .storageModeShared
        )
        print("Frustum points: \(frustumPointCount)")
    }
    
    struct IntegrateParams {
        var view: float4x4         // 64
        var proj: float4x4         // 64
        var viewInv: float4x4      // 64
        var projInv: float4x4      // 64
        var voxCount: SIMD3<UInt32> // 16 (SIMD3 is stored as SIMD4 in Swift)
        var voxSize: Float         // 4
        var voxDist: Float         // 4
        var voxMin: Float          // 4
        var depthDispThresh: Float // 4
        var numPlayers: Int32      // 4
        var _pad0: Int32 = 0       // 4
        var _pad1: Int64 = 0       // 8  → total 304
    }

    func clearVolume() {
        guard let volume = volumeTexture else { return }

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(clearVolumePipeline)
        encoder.setTexture(volume, index: 0)

        let w = clearVolumePipeline.threadExecutionWidth
        let groupSize = MTLSize(width: w, height: 1, depth: 1)
        let gridSize = MTLSize(width: volume.width, height: volume.height, depth: volume.depth)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func integrateDepth(depthTexture: MTLTexture, normTexture: MTLTexture,
                        dilatedDepth: MTLTexture, frame: DepthFrame) {
        guard let volume = volumeTexture, let frustum = frustumBuffer else { return }

        var params = IntegrateParams(
            view: frame.view[0],
            proj: frame.proj[0],
            viewInv: frame.viewInv[0],
            projInv: frame.projInv[0],
            voxCount: SIMD3<UInt32>(UInt32(frame.voxelCount.x), UInt32(frame.voxelCount.y), UInt32(frame.voxelCount.z)),
            voxSize: frame.voxelSize,
            voxDist: frame.voxelDist,
            voxMin: 0.1,
            depthDispThresh: 1.0,
            numPlayers: frame.numPlayers
        )

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(integratePipeline)

        encoder.setBuffer(frustum, offset: 0, index: 0)
        encoder.setBytes(&params, length: MemoryLayout<IntegrateParams>.size, index: 1)
        encoder.setTexture(depthTexture, index: 0)
        encoder.setTexture(normTexture, index: 1)
        encoder.setTexture(dilatedDepth, index: 2)
        encoder.setTexture(volume, index: 3)

        let w = integratePipeline.threadExecutionWidth
        let gridSize = MTLSize(width: frustumPointCount, height: 1, depth: 1)
        let groupSize = MTLSize(width: w, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    struct NormParams {
        var projInv: float4x4
        var viewInv: float4x4
        var texSize: SIMD2<UInt32>
        var _pad: SIMD2<UInt32> = .zero  // 8 bytes padding → 144 total
    }

    func generateNormals(depthTexture: MTLTexture, frame: DepthFrame) -> MTLTexture {
        let w = depthTexture.width
        let h = depthTexture.height

        if normalsTexture == nil || normalsTexture!.width != w || normalsTexture!.height != h {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba32Float,
                width: w, height: h,
                mipmapped: false
            )
            desc.usage = [.shaderRead, .shaderWrite]
            desc.storageMode = .private
            normalsTexture = device.makeTexture(descriptor: desc)!
        }

        var params = NormParams(
            projInv: frame.projInv[0],
            viewInv: frame.viewInv[0],
            texSize: SIMD2<UInt32>(UInt32(w), UInt32(h))
        )

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(normalsPipeline)
        encoder.setTexture(depthTexture, index: 0)
        encoder.setTexture(normalsTexture!, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<NormParams>.size, index: 0)

        let gW = normalsPipeline.threadExecutionWidth
        let gH = normalsPipeline.maxTotalThreadsPerThreadgroup / gW
        encoder.dispatchThreads(
            MTLSize(width: w, height: h, depth: 1),
            threadsPerThreadgroup: MTLSize(width: gW, height: gH, depth: 1)
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return normalsTexture!
    }
    
    func countNonEmptyVoxels() -> (count: Int, samples: [Float]) {
        guard let volume = volumeTexture else { return (0, []) }

        let counterBuffer = device.makeBuffer(length: 4, options: .storageModeShared)!
        counterBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)

        // We need a small kernel for this — let's read back a slice instead
        // For now, read a small region of the volume to check
        let sliceSize = 64
        let bytesPerRow = sliceSize * 4
        let bytesPerImage = bytesPerRow * sliceSize

        var data = [Float](repeating: 0, count: sliceSize * sliceSize * sliceSize)

        // Read center of volume
        let originX = volume.width / 2 - sliceSize / 2
        let originY = volume.height / 2 - sliceSize / 2
        let originZ = volume.depth / 2 - sliceSize / 2

        let region = MTLRegion(
            origin: MTLOrigin(x: originX, y: originY, z: originZ),
            size: MTLSize(width: sliceSize, height: sliceSize, depth: sliceSize)
        )

        volume.getBytes(
            &data,
            bytesPerRow: bytesPerRow,
            bytesPerImage: bytesPerImage,
            from: region,
            mipmapLevel: 0,
            slice: 0
        )

        var count = 0
        var sampleValues: [Float] = []
        for val in data {
            if val != -1.0 {
                count += 1
                if sampleValues.count < 5 { sampleValues.append(val) }
            }
        }
        return (count, sampleValues)
    }

    struct VoxelData {
        var coordX: UInt32
        var coordY: UInt32
        var coordZ: UInt32
        var value: Float
    }

    func extractNonEmptyVoxels() -> [VoxelData] {
        guard let volume = volumeTexture else { return [] }

        let maxOutput: UInt32 = 500_000
        let counterBuffer = device.makeBuffer(length: 4, options: .storageModeShared)!
        counterBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)

        let outputBuffer = device.makeBuffer(
            length: Int(maxOutput) * MemoryLayout<VoxelData>.stride,
            options: .storageModeShared
        )!

        var maxOutputVar = maxOutput

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(extractPipeline)
        encoder.setTexture(volume, index: 0)
        encoder.setBuffer(counterBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&maxOutputVar, length: 4, index: 2)

        let w = extractPipeline.threadExecutionWidth
        let gridSize = MTLSize(width: volume.width, height: volume.height, depth: volume.depth)
        let groupSize = MTLSize(width: w, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let count = Int(counterBuffer.contents().load(as: UInt32.self))
        let actualCount = min(count, Int(maxOutput))

        let ptr = outputBuffer.contents().bindMemory(to: VoxelData.self, capacity: actualCount)
        return Array(UnsafeBufferPointer(start: ptr, count: actualCount))
    }

    // Serialize and send back to Quest
    static func serializeVoxels(_ voxels: [MetalPipeline.VoxelData]) -> Data {
        var data = Data()

        // 4 bytes: voxel count
        var count = UInt32(voxels.count)
        data.append(Data(bytes: &count, count: 4))

        // Each voxel: 3x uint32 coord + 1x float value = 16 bytes
        for var voxel in voxels {
            data.append(Data(bytes: &voxel.coordX, count: 4))
            data.append(Data(bytes: &voxel.coordY, count: 4))
            data.append(Data(bytes: &voxel.coordZ, count: 4))
            data.append(Data(bytes: &voxel.value, count: 4))
        }

        return data
    }

}
