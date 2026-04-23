import Foundation
import Metal
import simd

// ─────────────────────────────────────────────────────────────────────────────
// MetalPipeline.swift — The Mac-side GPU pipeline manager
//
// WHAT THIS FILE DOES:
//   This class owns every Metal GPU resource (textures, buffers, pipelines)
//   and exposes one clean method per processing stage. It's the bridge between
//   the raw bytes arriving from the Quest and the triangle mesh that gets
//   sent back.
//
// LIFECYCLE:
//   One MetalPipeline is created at startup (EdgeMetalServer.metal is a singleton).
//   The same instance handles every frame — heavy resources like the TSDF volume
//   and frustum buffer are created once and reused forever.
//
// PROCESSING STEPS (called in order per frame):
//   1. uploadDepthTexture  — copies raw depth pixels into a GPU texture
//   2. dilateDepth         — fills holes in the depth image (8 jump-flood passes)
//   3. generateNormals     — estimates surface normals from depth gradients
//   4. setupVolume         — creates the 3D TSDF texture (first frame only)
//   5. setupFrustum        — pre-computes frustum voxel list (first frame only)
//   6. integrateDepth      — fuses this frame into the TSDF volume
//   7. generateMesh        — runs Surface Nets to extract a triangle mesh
// ─────────────────────────────────────────────────────────────────────────────

class MetalPipeline: @unchecked Sendable {

    // ── Metal device and command queue ───────────────────────────────────────
    // The MTLDevice represents the GPU (M3 Max in our case).
    // The MTLCommandQueue is the submission point — we build command buffers
    // and enqueue them here to run on the GPU.
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    // ── Compute pipeline states ───────────────────────────────────────────────
    // Each MTLComputePipelineState wraps one GPU kernel function.
    // They are compiled once at init and reused for every frame.
    let depthStatsPipeline: MTLComputePipelineState   // (unused in main pipeline — debug helper)
    let initDilationPipeline: MTLComputePipelineState  // Copies depth into ping-pong buffer
    let dilatePipeline: MTLComputePipelineState        // One dilation pass (called 8 times)

    // ── Dilation ping-pong textures ───────────────────────────────────────────
    // Jump-flood dilation reads from texA and writes to texB, then swaps them.
    // Both are RGBA32Float: (original_x, original_y, depth_NDC, unused)
    var dilationA: MTLTexture?
    var dilationB: MTLTexture?

    // ── TSDF volume (the 3D reconstruction grid) ──────────────────────────────
    // RG32Float 3D texture: R = TSDF value, G = weight (observation count)
    // Created once, persists across all frames — this IS the 3D map of the room.
    var volumeTexture: MTLTexture?

    // ── Frustum voxel list ────────────────────────────────────────────────────
    // Pre-computed list of view-space positions that lie inside the camera frustum.
    // The `integrate` kernel runs one thread per position in this list.
    // Created once (frustum shape doesn't change frame-to-frame).
    var frustumBuffer: MTLBuffer?
    var frustumPointCount: Int = 0

    let clearVolumePipeline: MTLComputePipelineState   // Resets all voxels to unobserved
    let integratePipeline: MTLComputePipelineState     // Fuses one depth frame into TSDF

    // ── Normal estimation ─────────────────────────────────────────────────────
    // Normals are computed per-pixel from the depth image and stored here.
    // The integrate kernel reads normals to weight observations by surface angle.
    let normalsPipeline: MTLComputePipelineState
    var normalsTexture: MTLTexture?

    // ── Extract pipeline (debug / legacy) ────────────────────────────────────
    // Used by extractNonEmptyVoxels() to pull voxel data back to the CPU.
    // No longer called in the main pipeline (replaced by camera-centred bounding box).
    let extractPipeline: MTLComputePipelineState

    // ── Surface Nets meshing ──────────────────────────────────────────────────
    // Two-pass mesh extraction:
    //   vertexPipeline → finds surface crossings and places one vertex per active cell
    //   indexPipeline  → connects neighbouring vertices into quads/triangles
    let vertexPipeline: MTLComputePipelineState
    let indexPipeline: MTLComputePipelineState

    // ─────────────────────────────────────────────────────────────────────────
    // init — Finds the GPU, compiles all shader functions into pipeline states
    //
    // Uses try! / fatalError intentionally — if Metal isn't available or a
    // shader fails to compile, there's no point running the server at all.
    // ─────────────────────────────────────────────────────────────────────────
    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device found")
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        // Load the compiled Metal library (all .metal files in the Sources bundle)
        let library = try! device.makeDefaultLibrary(bundle: Bundle.module)

        // Compile each kernel function into a compute pipeline state
        // (This is like linking a shader program — done once at startup)
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

        let vertFunc = library.makeFunction(name: "surfaceNetsVertices")!
        self.vertexPipeline = try! device.makeComputePipelineState(function: vertFunc)

        let idxFunc = library.makeFunction(name: "surfaceNetsIndices")!
        self.indexPipeline = try! device.makeComputePipelineState(function: idxFunc)

        print("Metal device: \(device.name)")
        // Debugging: confirm struct size matches what the Metal shader expects
        print("IntegrateParams size: \(MemoryLayout<IntegrateParams>.size), stride: \(MemoryLayout<IntegrateParams>.stride)")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // uploadDepthTexture — Copies raw depth pixels from CPU memory to a GPU texture
    //
    // The Quest sends depth as an array of 32-bit floats (one float per pixel).
    // We create an R32Float (single-channel float) texture and copy the bytes in.
    // The texture is then used as input to dilation, normals, and integration.
    // ─────────────────────────────────────────────────────────────────────────
    func uploadDepthTexture(frame: DepthFrame) -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,    // Single channel — just the depth value
            width: frame.width,
            height: frame.height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]  // Only read by shaders, never written

        let texture = device.makeTexture(descriptor: descriptor)!

        // Copy the raw float bytes directly into the texture
        frame.depthPixels.withUnsafeBytes { ptr in
            texture.replace(
                region: MTLRegionMake2D(0, 0, frame.width, frame.height),
                mipmapLevel: 0,
                withBytes: ptr.baseAddress!,
                bytesPerRow: frame.width * 4   // 4 bytes per float
            )
        }

        return texture
    }

    // ─────────────────────────────────────────────────────────────────────────
    // computeDepthStats — (Debug helper) Gets min/max depth values from a texture
    //
    // Runs a GPU kernel that atomically finds the min and max float values.
    // Values are encoded as scaled integers (×10000) because Metal atomics only
    // work on integers. Not used in the main pipeline — useful for debugging.
    // ─────────────────────────────────────────────────────────────────────────
    func computeDepthStats(texture: MTLTexture) -> (min: Float, max: Float) {
        let minBuffer = device.makeBuffer(length: 4, options: .storageModeShared)!
        let maxBuffer = device.makeBuffer(length: 4, options: .storageModeShared)!

        // Init min to large value (0xFFFFFFFF), max to 0
        minBuffer.contents().storeBytes(of: UInt32(0xFFFFFFFF), as: UInt32.self)
        maxBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(depthStatsPipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(minBuffer, offset: 0, index: 0)
        encoder.setBuffer(maxBuffer, offset: 0, index: 1)

        // Choose thread group size to fill GPU efficiently
        let w = depthStatsPipeline.threadExecutionWidth
        let h = depthStatsPipeline.maxTotalThreadsPerThreadgroup / w
        let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
        let gridSize = MTLSize(width: texture.width, height: texture.height, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Decode the scaled integers back to floats
        let minInt = minBuffer.contents().load(as: UInt32.self)
        let maxInt = maxBuffer.contents().load(as: UInt32.self)
        return (Float(minInt) / 10000.0, Float(maxInt) / 10000.0)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DilationParams — Parameters passed to the dilation kernel
    //
    // This struct is packed carefully to match the Metal shader layout.
    // The padding fields (_pad1, _pad2) are here because Metal requires structs
    // to be 16-byte aligned — without them the shader would read wrong bytes.
    // ─────────────────────────────────────────────────────────────────────────
    struct DilationParams {
        var proj: float4x4          // 64 bytes — camera projection (for depth-to-metres conversion)
        var texSize: SIMD2<UInt32>  // 8 bytes  — depth image dimensions in pixels
        var voxDist: Float          // 4 bytes  — TSDF truncation distance (how far to dilate)
        var voxSize: Float          // 4 bytes  — voxel size (controls acceptance radius)
        var stepSize: Int32         // 4 bytes  — current jump-flood step in pixels
        var _pad1: Int32 = 0        // 4 bytes padding
        var _pad2: Int64 = 0        // 8 bytes padding → total 96 bytes (64-byte aligned)
    }

    // Creates an RGBA32Float texture suitable for ping-pong dilation buffers
    // RGBA = (source_x, source_y, depth_ndc, unused) — track where each depth came from
    func createDilationTexture(width: Int, height: Int) -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]  // Read from one, write to the other
        desc.storageMode = .private               // GPU-only, fastest access
        return device.makeTexture(descriptor: desc)!
    }

    // ─────────────────────────────────────────────────────────────────────────
    // dilateDepth — Fills holes in the depth image using jump-flood morphological dilation
    //
    // WHAT IT DOES:
    //   The raw depth image from the Quest has gaps (zero pixels) wherever the
    //   sensor couldn't measure depth (reflective surfaces, edges, thin objects).
    //   This function spreads valid depth values outward into those gaps.
    //
    // HOW IT WORKS (jump-flood):
    //   Instead of one pass checking every pixel, we run `dilationSteps` passes.
    //   Each pass samples neighbours at distance `stepSize` pixels, where stepSize
    //   halves every pass: 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 (→ 1)
    //   This fills gaps up to 256 pixels wide in just 8 passes.
    //
    //   Two textures (dilationA, dilationB) are used as ping-pong buffers so
    //   reads and writes don't conflict. After each pass they are swapped.
    //
    // ACCEPTANCE RULE:
    //   A gap pixel only adopts a neighbour's depth if it's:
    //   - within the physically meaningful fill radius (based on actual depth)
    //   - closer than what the pixel currently holds (nearer depth wins)
    //
    // Returns the texture that contains the final dilated depth.
    // ─────────────────────────────────────────────────────────────────────────
    func dilateDepth(depthTexture: MTLTexture, frame: DepthFrame,
                     voxelSize: Float = 0.1, voxelDist: Float = 0.2,
                     dilationSteps: Int = 8) -> MTLTexture {
        let w = depthTexture.width
        let h = depthTexture.height

        // Lazily create (or recreate if size changed) the ping-pong textures
        if dilationA == nil || dilationA!.width != w || dilationA!.height != h {
            dilationA = createDilationTexture(width: w, height: h)
            dilationB = createDilationTexture(width: w, height: h)
        }

        let commandBuffer = commandQueue.makeCommandBuffer()!

        // ── Pass 0: Init — copy raw depth into dilationA in the (x, y, depth, 0) format ──
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

        // ── Passes 1..dilationSteps: each one halves the step size ───────────
        // maxStep = 2^dilationSteps (e.g. 8 steps → starts at 256)
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
            encoder.setTexture(texA, index: 0)   // Read from texA
            encoder.setTexture(texB, index: 1)   // Write to texB
            encoder.setBytes(&params, length: MemoryLayout<DilationParams>.size, index: 0)

            let dGroupW = dilatePipeline.threadExecutionWidth
            let dGroupH = dilatePipeline.maxTotalThreadsPerThreadgroup / dGroupW
            encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: dGroupW, height: dGroupH, depth: 1))
            encoder.endEncoding()

            stepSize /= 2
            swap(&texA, &texB)  // Next pass reads what we just wrote
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // After the last swap, texA holds the final result
        return texA
    }

    // ─────────────────────────────────────────────────────────────────────────
    // setupVolume — Creates the TSDF 3D texture (runs only once, on the first frame)
    //
    // The volume is a 3D texture where each voxel stores:
    //   R channel = TSDF value (signed distance to nearest surface, normalised to [-1, 1])
    //   G channel = weight (how many times this voxel has been observed, 0 = never seen)
    //
    // Format: RG32Float (two 32-bit floats per voxel)
    // StorageMode: managed (visible to both CPU and GPU — needed for debugging readbacks)
    //
    // The volume is persistent — it accumulates observations across many frames,
    // gradually building up a stable 3D map of the room.
    // ─────────────────────────────────────────────────────────────────────────
    func setupVolume(frame: DepthFrame) {
        if volumeTexture != nil { return }  // Already exists — nothing to do

        let desc = MTLTextureDescriptor()
        desc.textureType = .type3D
        desc.pixelFormat = .rg32Float
        desc.width  = Int(frame.voxelCount.x)
        desc.height = Int(frame.voxelCount.y)
        desc.depth  = Int(frame.voxelCount.z)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .managed  // CPU-accessible (for debug readbacks)

        volumeTexture = device.makeTexture(descriptor: desc)!
        print("Created volume: \(volumeTexture!.width)x\(volumeTexture!.height)x\(volumeTexture!.depth)")

        // Reset all voxels to "never observed" (weight = 0)
        clearVolume()
        print("Volume cleared")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // setupFrustum — Pre-computes the list of voxel positions inside the camera frustum
    //               (runs only once, on the first frame)
    //
    // WHY WE NEED THIS:
    //   The `integrate` kernel needs to know which voxels are inside the camera's
    //   field of view for this frame. Computing that per-frame is expensive.
    //   Since the frustum shape (the camera's FOV cone) doesn't change, we
    //   pre-compute all possible view-space positions once and reuse the list.
    //
    // HOW IT WORKS:
    //   1. Use the inverse projection matrix to find the 4 corner rays of the frustum.
    //   2. For each depth slice (z from near to maxDist in steps of voxelSize):
    //      compute the X and Y bounds of the frustum at that depth.
    //   3. Step through all voxel-sized positions within those bounds.
    //   4. Store each position as a view-space SIMD3<Float> in the frustumBuffer.
    //
    // The buffer typically holds ~200,000 positions — one GPU thread per position
    // during integration. This is much faster than iterating all 2M voxels.
    // ─────────────────────────────────────────────────────────────────────────
    func setupFrustum(frame: DepthFrame) {
        if frustumBuffer != nil { return }  // Already set up

        let proj = frame.proj[0]
        let voxelSize = frame.voxelSize
        let maxDist = frame.maxUpdateDist
        let minDist: Float = 0.1  // Skip voxels very close to the camera lens

        // Use inverse projection to find where the frustum corners are at z=1
        let projInv = frame.projInv[0]

        // Unproject a clip-space corner ray to view-space direction
        func unprojectCorner(ndcX: Float, ndcY: Float) -> SIMD3<Float> {
            let clip = SIMD4<Float>(ndcX, ndcY, -1, 1)
            let view = projInv * clip
            return SIMD3<Float>(view.x, view.y, view.z) / view.w
        }

        // Get the 4 frustum corners (bottom-left, bottom-right, top-left, top-right)
        let bl = unprojectCorner(ndcX: -1, ndcY: -1)
        let br = unprojectCorner(ndcX:  1, ndcY: -1)
        let tl = unprojectCorner(ndcX: -1, ndcY:  1)
        // tr = unprojectCorner(ndcX: 1, ndcY: 1) — not needed, derived from br and tl

        // Slopes: how far in X or Y per unit of depth (Z)
        // These define the widening angle of the frustum
        let leftSlope   = bl.x / (-bl.z)
        let rightSlope  = br.x / (-br.z)
        let bottomSlope = bl.y / (-bl.z)
        let topSlope    = tl.y / (-tl.z)

        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(200_000)

        // Walk through every voxel-sized step from near to maxDist
        let near = abs(proj[2][3] * 0.5)  // Near plane distance from projection matrix
        var z: Float = near
        while z < maxDist {
            // At depth z, the frustum spans this X and Y range
            let xMin = leftSlope   * z + voxelSize
            let xMax = rightSlope  * z - voxelSize
            let yMin = bottomSlope * z + voxelSize
            let yMax = topSlope    * z - voxelSize

            var x = xMin
            while x < xMax {
                var y = yMin
                while y < yMax {
                    let v = SIMD3<Float>(x, y, -z)  // Negative Z = forward in view space
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
        // Upload to a shared buffer (CPU writes once, GPU reads every frame)
        frustumBuffer = device.makeBuffer(
            bytes: positions,
            length: positions.count * MemoryLayout<SIMD3<Float>>.stride,
            options: .storageModeShared
        )
        print("Frustum points: \(frustumPointCount)")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IntegrateParams — Parameters passed to the integrate Metal kernel
    //
    // Padding fields are required to satisfy Metal's 16-byte struct alignment rule.
    // If padding is wrong, the shader reads the values from the wrong memory offsets.
    // ─────────────────────────────────────────────────────────────────────────
    struct IntegrateParams {
        var view: float4x4         // 64 bytes — world → camera space transform
        var proj: float4x4         // 64 bytes — camera → clip space projection
        var viewInv: float4x4      // 64 bytes — camera → world space (for eye position)
        var projInv: float4x4      // 64 bytes — clip → camera space (for unprojection)
        var voxCount: SIMD3<UInt32> // 16 bytes — volume grid dimensions (SIMD3 stored as SIMD4)
        var voxSize: Float         // 4 bytes  — voxel size in metres
        var voxDist: Float         // 4 bytes  — TSDF truncation distance in metres
        var voxMin: Float          // 4 bytes  — minimum distance from camera to integrate
        var depthDispThresh: Float // 4 bytes  — accepted depth disparity for occlusion check
        var numPlayers: Int32      // 4 bytes  — number of valid entries in the player heads buffer
        var _pad0: Int32 = 0       // 4 bytes padding
        var _pad1: Int64 = 0       // 8 bytes padding → total 304 bytes
    }

    // ── Per-frame player head buffer ──────────────────────────────────────────
    // 8 × float3 world positions, uploaded each frame.
    // Reused across frames to avoid allocating a buffer every frame.
    var playerHeadsBuffer: MTLBuffer?

    // ─────────────────────────────────────────────────────────────────────────
    // clearVolume — Resets all voxels in the TSDF volume to "unobserved"
    //
    // Called once when the volume is first created. After this call, every voxel
    // has weight = 0, meaning it has never been seen by the depth sensor.
    // ─────────────────────────────────────────────────────────────────────────
    func clearVolume() {
        guard let volume = volumeTexture else { return }

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(clearVolumePipeline)
        encoder.setTexture(volume, index: 0)

        // Dispatch one thread per voxel
        let w = clearVolumePipeline.threadExecutionWidth
        let groupSize = MTLSize(width: w, height: 1, depth: 1)
        let gridSize = MTLSize(width: volume.width, height: volume.height, depth: volume.depth)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // integrateDepth — Fuses one depth frame into the TSDF volume
    //
    // This is the core reconstruction step. One GPU thread per frustum voxel.
    //
    // For each voxel position (from the pre-built frustum list):
    //   1. Transform from view space → world space using the current camera pose
    //   2. Find which voxel cell that world position maps to
    //   3. Project that voxel onto the depth image to find the measured depth pixel
    //   4. Compute: signed distance = measured_depth - voxel_distance_from_camera
    //      (positive = voxel is in front of the surface, negative = behind)
    //   5. Blend this new reading into the existing TSDF value using weighted average
    //      (weight capped at 30, so old data doesn't dominate forever)
    //
    // Quality gates: readings are rejected if:
    //   - Outside the truncation band (too far from any surface)
    //   - Surface normal is near-perpendicular to the view ray (grazing angle)
    //   - Occluded by nearer geometry (dilation occlusion check)
    // ─────────────────────────────────────────────────────────────────────────
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
            voxMin: 0.1,              // Minimum depth to integrate (100mm from lens)
            depthDispThresh: 1.0,     // Occlusion tolerance — how much dilation gap is OK
            numPlayers: frame.numPlayers
        )

        // ── Upload player head positions to a GPU buffer ──────────────────────
        // MAX_PLAYERS × SIMD3<Float> (stored as 16 bytes each due to SIMD alignment)
        // We flatten to float3 explicitly using a tightly packed float array
        // to match the shader's `constant float3*` binding.
        let heads = frame.playerHeads
        // Pack as contiguous floats (x0,y0,z0, x1,y1,z1, ...) — 12 bytes per head
        var packed = [Float]()
        packed.reserveCapacity(MAX_PLAYERS * 3)
        for h in heads {
            packed.append(h.x); packed.append(h.y); packed.append(h.z)
        }
        // Pad to MAX_PLAYERS if frame had fewer entries (should always be 8 from Quest)
        while packed.count < MAX_PLAYERS * 3 { packed.append(0) }

        let headsBytes = packed.count * MemoryLayout<Float>.size
        if playerHeadsBuffer == nil || playerHeadsBuffer!.length < headsBytes {
            playerHeadsBuffer = device.makeBuffer(length: headsBytes, options: .storageModeShared)!
        }
        memcpy(playerHeadsBuffer!.contents(), packed, headsBytes)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(integratePipeline)

        // Buffer 0: frustum positions (view-space voxel centres)
        encoder.setBuffer(frustum, offset: 0, index: 0)
        // Buffer 1: camera matrices and volume config
        encoder.setBytes(&params, length: MemoryLayout<IntegrateParams>.size, index: 1)
        // Buffer 2: player head world positions for exclusion cylinders
        encoder.setBuffer(playerHeadsBuffer!, offset: 0, index: 2)
        // Textures: original depth, surface normals, dilated depth, TSDF volume (read+write)
        encoder.setTexture(depthTexture, index: 0)
        encoder.setTexture(normTexture,  index: 1)
        encoder.setTexture(dilatedDepth, index: 2)
        encoder.setTexture(volume,       index: 3)

        // One thread per frustum voxel
        let w = integratePipeline.threadExecutionWidth
        let gridSize = MTLSize(width: frustumPointCount, height: 1, depth: 1)
        let groupSize = MTLSize(width: w, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // NormParams — Parameters for the depth normals kernel
    //
    // The depthNormals kernel needs the inverse matrices to convert screen pixels
    // back into 3D world-space positions (so it can compute the 3D surface normal).
    // ─────────────────────────────────────────────────────────────────────────
    struct NormParams {
        var projInv: float4x4
        var viewInv: float4x4
        var texSize: SIMD2<UInt32>
        var _pad: SIMD2<UInt32> = .zero  // 8 bytes padding → 144 bytes total
    }

    // ─────────────────────────────────────────────────────────────────────────
    // generateNormals — Estimates surface normals from the depth image
    //
    // For each pixel, samples the 4 neighbouring pixels (left, right, up, down),
    // converts them to 3D world-space positions, and takes the cross product to
    // get the direction perpendicular to the local surface.
    //
    // NOTE: We use the original depth texture (not dilated) for normals.
    //   The dilated depth can "bleed" valid depth into depth-discontinuity areas,
    //   which would produce incorrect normals at object edges. The original depth
    //   gives cleaner, more accurate surface orientations.
    //
    // Output: RGBA32Float texture, RGB = normal direction, A = 1.0
    //   Pixels with missing neighbours get normal (0,0,0) = invalid.
    // ─────────────────────────────────────────────────────────────────────────
    func generateNormals(depthTexture: MTLTexture, frame: DepthFrame) -> MTLTexture {
        let w = depthTexture.width
        let h = depthTexture.height

        // Create (or recreate) the output normals texture if size changed
        if normalsTexture == nil || normalsTexture!.width != w || normalsTexture!.height != h {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba32Float,
                width: w, height: h,
                mipmapped: false
            )
            desc.usage = [.shaderRead, .shaderWrite]
            desc.storageMode = .private  // GPU-only — never read back to CPU
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
        encoder.setTexture(depthTexture,  index: 0)  // Input: depth image
        encoder.setTexture(normalsTexture!, index: 1)  // Output: normal map
        encoder.setBytes(&params, length: MemoryLayout<NormParams>.size, index: 0)

        // One thread per pixel
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

    // ─────────────────────────────────────────────────────────────────────────
    // countNonEmptyVoxels — (Debug helper) CPU readback of a 64³ centre slice
    //
    // Reads the middle portion of the volume back to the CPU and counts voxels
    // that have been observed (non-(-1.0) values). Not used in the main pipeline.
    // ─────────────────────────────────────────────────────────────────────────
    func countNonEmptyVoxels() -> (count: Int, samples: [Float]) {
        guard let volume = volumeTexture else { return (0, []) }

        // Read a 64×64×64 cube from the centre of the volume
        let sliceSize = 64
        let bytesPerRow = sliceSize * 4
        let bytesPerImage = bytesPerRow * sliceSize

        var data = [Float](repeating: 0, count: sliceSize * sliceSize * sliceSize)

        let originX = volume.width  / 2 - sliceSize / 2
        let originY = volume.height / 2 - sliceSize / 2
        let originZ = volume.depth  / 2 - sliceSize / 2

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

    // ─────────────────────────────────────────────────────────────────────────
    // VoxelData — A single observed voxel's coordinate and TSDF value
    // Used by extractNonEmptyVoxels (legacy GPU readback path)
    // ─────────────────────────────────────────────────────────────────────────
    struct VoxelData {
        var coordX: UInt32
        var coordY: UInt32
        var coordZ: UInt32
        var value: Float   // TSDF value at this voxel
    }

    // ─────────────────────────────────────────────────────────────────────────
    // extractNonEmptyVoxels — (Legacy) GPU kernel readback of all observed voxels
    //
    // Previously used to build the mesh region from actual observed voxel data.
    // Replaced by the camera-centred bounding box in generateMesh() because:
    //   - Reading 500k voxels back to the CPU every frame was very slow
    //   - Camera position gives an equally good (and much faster) region estimate
    //
    // Kept here as a reference / for potential future debug use.
    // ─────────────────────────────────────────────────────────────────────────
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
        encoder.setBuffer(outputBuffer,  offset: 0, index: 1)
        encoder.setBytes(&maxOutputVar, length: 4, index: 2)

        let w = extractPipeline.threadExecutionWidth
        let gridSize  = MTLSize(width: volume.width, height: volume.height, depth: volume.depth)
        let groupSize = MTLSize(width: w, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let count       = Int(counterBuffer.contents().load(as: UInt32.self))
        let actualCount = min(count, Int(maxOutput))

        let ptr = outputBuffer.contents().bindMemory(to: VoxelData.self, capacity: actualCount)
        return Array(UnsafeBufferPointer(start: ptr, count: actualCount))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // serializeVoxels — Serializes VoxelData array to bytes for sending to Quest
    // (Legacy — only used with extractNonEmptyVoxels, not in the current mesh pipeline)
    //
    // Format: 4-byte count + 16 bytes per voxel (3× uint32 coord + 1× float value)
    // ─────────────────────────────────────────────────────────────────────────
    static func serializeVoxels(_ voxels: [MetalPipeline.VoxelData]) -> Data {
        var data = Data()

        var count = UInt32(voxels.count)
        data.append(Data(bytes: &count, count: 4))

        for var voxel in voxels {
            data.append(Data(bytes: &voxel.coordX, count: 4))
            data.append(Data(bytes: &voxel.coordY, count: 4))
            data.append(Data(bytes: &voxel.coordZ, count: 4))
            data.append(Data(bytes: &voxel.value,  count: 4))
        }

        return data
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MeshParams — Parameters passed to both Surface Nets GPU kernels
    // ─────────────────────────────────────────────────────────────────────────
    struct MeshParams {
        var voxCount:  SIMD3<Int32>   // Full volume dimensions (e.g. 128, 128, 128)
        var voxSize:   Float          // Size of each voxel in metres
        var regionMin: SIMD3<Int32>   // First voxel coordinate of the meshing region
        var regionMax: SIMD3<Int32>   // Last voxel coordinate of the meshing region
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MeshResult — The triangle mesh returned from generateMesh
    //
    // vertices: flat array of floats, 6 floats per vertex:
    //   [px, py, pz, nx, ny, nz]  (position XYZ + normal XYZ)
    //   This is packed as packed_float3 pairs in the Metal shader (24 bytes/vertex)
    //
    // indices: triangle index list — every 3 entries form one triangle
    // vertexCount / triangleCount: counts for logging and sending
    // ─────────────────────────────────────────────────────────────────────────
    struct MeshResult {
        var vertices:      [Float]    // Packed position + normal floats
        var indices:       [UInt32]   // Triangle indices
        var vertexCount:   Int
        var triangleCount: Int
    }

    // ── coordVertMap buffer ────────────────────────────────────────────────────
    // Shared between the two Surface Nets passes:
    //   Pass 1 writes: voxel index → vertex index (-1 if no vertex)
    //   Pass 2 reads:  looks up the 4 surrounding vertices to form a quad
    // Reused across frames to avoid reallocating the large buffer every time.
    var coordVertMapBuffer: MTLBuffer?

    // ─────────────────────────────────────────────────────────────────────────
    // generateMesh — Runs Surface Nets to extract a triangle mesh from the TSDF volume
    //
    // TWO-PASS GPU ALGORITHM:
    //   Pass 1 (surfaceNetsVertices):
    //     For each voxel cell in the meshing region, checks if the TSDF crosses
    //     zero (positive → negative or vice versa). If ≥3 edge crossings are found,
    //     places a vertex at the average crossing position and records it in coordVertMap.
    //
    //   Pass 2 (surfaceNetsIndices):
    //     For each voxel cell that has a vertex, looks at the 3 axis-aligned edges.
    //     Where there's a surface crossing, forms a quad (2 triangles) from the 4
    //     surrounding cells. Winding order is determined by which side is "inside".
    //
    // CAMERA-CENTRED BOUNDING BOX (key optimization):
    //   Instead of meshing the entire 128³ volume (2M voxels → crash), we only mesh
    //   a cube centred around the camera's current position. The cube extends
    //   `maxUpdateDist / voxelSize + 2` voxels in each direction.
    //   This is computed purely from frame.viewInv (no GPU readback needed), making
    //   it fast even when the previous approach (extractNonEmptyVoxels) was slow.
    //
    //   eyeCol = frame.viewInv[0][3]  ← 4th column of the inverse view matrix = camera position
    //   eyeVox = camera position in voxel coordinates
    //   regionMin/Max = eyeVox ± extentVox, clamped to volume bounds
    // ─────────────────────────────────────────────────────────────────────────
    func generateMesh(frame: DepthFrame) -> MeshResult {
        guard let volume = volumeTexture else {
            return MeshResult(vertices: [], indices: [], vertexCount: 0, triangleCount: 0)
        }

        let voxCount = SIMD3<Int32>(frame.voxelCount.x, frame.voxelCount.y, frame.voxelCount.z)
        let totalVoxels = Int(voxCount.x) * Int(voxCount.y) * Int(voxCount.z)

        // ── Camera-centred bounding box ───────────────────────────────────────
        // Extract camera world position from the 4th column of the inverse view matrix
        // (viewInv transforms camera → world, so its translation column = world eye pos)
        let eyeCol   = frame.viewInv[0][3]
        let eyeWorld = SIMD3<Float>(eyeCol.x, eyeCol.y, eyeCol.z)

        // Convert camera world position to voxel grid coordinates
        // Formula: (world_pos / voxelSize) + (voxelCount / 2)  — volume is centred at origin
        let extentVox = Int32(frame.maxUpdateDist / frame.voxelSize) + 2
        let eyeVox = SIMD3<Int32>(
            Int32(eyeWorld.x / frame.voxelSize + Float(voxCount.x) * 0.5),
            Int32(eyeWorld.y / frame.voxelSize + Float(voxCount.y) * 0.5),
            Int32(eyeWorld.z / frame.voxelSize + Float(voxCount.z) * 0.5)
        )

        // Mesh only the cube around the camera, clamped to volume bounds
        let regionMin   = max(eyeVox &- extentVox, SIMD3<Int32>(0, 0, 0))
        let regionMax   = min(eyeVox &+ extentVox, voxCount)
        let regionSize  = regionMax &- regionMin
        let regionTotal = Int(regionSize.x) * Int(regionSize.y) * Int(regionSize.z)
        print("Mesh region: \(regionMin) to \(regionMax), size \(regionSize), total \(regionTotal)")

        // Buffer size estimates based on the region (not full volume)
        // maxVertices: at most 1 vertex per 3 voxels (heuristic)
        // maxTriIndices: at most 6 indices (2 triangles) per voxel per axis = 18 per voxel,
        //                but only a fraction of voxels are on the surface, so 6× is safe
        let maxVertices   = regionTotal / 3
        let maxTriIndices = regionTotal * 6

        // ── coordVertMap: voxel-to-vertex lookup table ────────────────────────
        // Size = totalVoxels (full volume) because indices use absolute voxel coords
        // Initialise to 0xFF (all bytes) = -1 (0xFFFFFFFF as int32 = INVALID_VERTEX)
        if coordVertMapBuffer == nil || coordVertMapBuffer!.length < totalVoxels * 4 {
            coordVertMapBuffer = device.makeBuffer(length: totalVoxels * 4, options: .storageModeShared)!
        }
        memset(coordVertMapBuffer!.contents(), 0xFF, totalVoxels * 4)

        // ── Output buffers ────────────────────────────────────────────────────
        // vertexBuffer: 24 bytes per vertex (packed_float3 pos + packed_float3 norm)
        let vertexBuffer  = device.makeBuffer(length: max(maxVertices * 24, 24), options: .storageModeShared)!
        let vertCountBuffer = device.makeBuffer(length: 4, options: .storageModeShared)!
        let triBuffer     = device.makeBuffer(length: max(maxTriIndices * 4, 4), options: .storageModeShared)!
        let triCountBuffer  = device.makeBuffer(length: 4, options: .storageModeShared)!

        // Zero-initialise the atomic counters
        vertCountBuffer.contents().storeBytes(of: Int32(0), as: Int32.self)
        triCountBuffer.contents().storeBytes(of: Int32(0), as: Int32.self)

        var params = MeshParams(
            voxCount:  voxCount,
            voxSize:   frame.voxelSize,
            regionMin: regionMin,
            regionMax: regionMax
        )
        print("MeshParams size: \(MemoryLayout<MeshParams>.size), stride: \(MemoryLayout<MeshParams>.stride)")

        let commandBuffer = commandQueue.makeCommandBuffer()!

        // ── Pass 1: Generate vertices ─────────────────────────────────────────
        // One thread per voxel cell in the meshing region
        let vertEncoder = commandBuffer.makeComputeCommandEncoder()!
        vertEncoder.setComputePipelineState(vertexPipeline)
        vertEncoder.setTexture(volume,                index: 0)  // TSDF volume (read-only)
        vertEncoder.setBuffer(vertexBuffer,           offset: 0, index: 0)  // Output: vertices
        vertEncoder.setBuffer(vertCountBuffer,        offset: 0, index: 1)  // Atomic vertex counter
        vertEncoder.setBuffer(coordVertMapBuffer!,    offset: 0, index: 2)  // voxel → vertex map
        vertEncoder.setBytes(&params, length: MemoryLayout<MeshParams>.size, index: 3)

        let w1 = vertexPipeline.threadExecutionWidth
        vertEncoder.dispatchThreads(
            MTLSize(width: regionTotal, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w1, height: 1, depth: 1)
        )
        vertEncoder.endEncoding()

        // ── Pass 2: Generate triangle indices ─────────────────────────────────
        // One thread per voxel cell in the meshing region
        // Reads coordVertMap (written by Pass 1) to find neighbouring vertices
        let idxEncoder = commandBuffer.makeComputeCommandEncoder()!
        idxEncoder.setComputePipelineState(indexPipeline)
        idxEncoder.setTexture(volume,             index: 0)  // TSDF volume (for crossing checks)
        idxEncoder.setBuffer(coordVertMapBuffer!, offset: 0, index: 0)  // voxel → vertex map
        idxEncoder.setBuffer(triBuffer,           offset: 0, index: 1)  // Output: triangle indices
        idxEncoder.setBuffer(triCountBuffer,      offset: 0, index: 2)  // Atomic triangle counter
        idxEncoder.setBytes(&params, length: MemoryLayout<MeshParams>.size, index: 3)
        idxEncoder.setBuffer(vertexBuffer,        offset: 0, index: 4)  // Vertices (not written to here)

        let w2 = indexPipeline.threadExecutionWidth
        idxEncoder.dispatchThreads(
            MTLSize(width: regionTotal, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w2, height: 1, depth: 1)
        )
        idxEncoder.endEncoding()

        // Both passes encode into the same command buffer — they run sequentially on GPU
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // ── Read results back to CPU ──────────────────────────────────────────
        let vertCount = Int(vertCountBuffer.contents().load(as: Int32.self))
        // Clamp triangle count to buffer capacity (safety guard against overflow)
        let triCount  = min(Int(triCountBuffer.contents().load(as: Int32.self)), maxTriIndices)

        // Read float data: 6 floats per vertex (pos.xyz, norm.xyz)
        let vertPtr = vertexBuffer.contents().bindMemory(to: Float.self, capacity: vertCount * 6)
        let verts   = Array(UnsafeBufferPointer(start: vertPtr, count: vertCount * 6))

        // Read index data: one UInt32 per index, 3 indices per triangle
        let idxPtr  = triBuffer.contents().bindMemory(to: UInt32.self, capacity: triCount)
        let indices = Array(UnsafeBufferPointer(start: idxPtr, count: triCount))

        return MeshResult(
            vertices:      verts,
            indices:       indices,
            vertexCount:   vertCount,
            triangleCount: triCount / 3
        )
    }
}
