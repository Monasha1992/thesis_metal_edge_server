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
//   2. dilateDepth         — fills holes in the depth image (2 jump-flood passes by default)
//   3. generateNormals     — estimates surface normals from depth gradients
//   4. setupVolume         — creates the 3D TSDF texture (first frame only)
//   5. setupFrustum        — pre-computes frustum voxel list (first frame only)
//   6. integrateDepth      — fuses this frame into the TSDF volume
//   7. generateChunkMeshes — runs Surface Nets per chunk (or generateMesh in
//                            legacy single-mesh mode) to extract triangles
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
    let dilatePipeline: MTLComputePipelineState        // One dilation pass (called dilationSteps times, default 2)

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
        let library: MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            print("Warning: Default Metal library not found in bundle. Attempting to compile from source...")
            // Fallback: Compile from .metal source files in the bundle
            var source = ""
            let shaderFiles = ["DepthDilation", "DepthNormal", "DepthProcess", "SurfaceNets", "VolumeIntegration"]
            for file in shaderFiles {
                if let url = Bundle.module.url(forResource: file, withExtension: "metal", subdirectory: "Shaders"),
                   let content = try? String(contentsOf: url) {
                    source += content + "\n"
                } else if let url = Bundle.module.url(forResource: file, withExtension: "metal"),
                          let content = try? String(contentsOf: url) {
                    source += content + "\n"
                }
            }
            
            if source.isEmpty {
                fatalError("Could not find any Metal shader sources in bundle.")
            }
            
            do {
                library = try device.makeLibrary(source: source, options: nil)
                print("Successfully compiled Metal library from source.")
            } catch {
                fatalError("Failed to compile Metal library from source: \(error)")
            }
        }

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
    // The Quest sends depth as uint16 normalized-NDC values (one per pixel, 2
    // bytes, little-endian). We create an .r16Unorm texture and copy the bytes
    // in; Metal samples .r16Unorm back as a normalized float in [0,1], so every
    // downstream shader (dilation, normals, integration) reads the same depth
    // value it did when depth was sent as float32 — no shader maths change.
    // ─────────────────────────────────────────────────────────────────────────
    func uploadDepthTexture(frame: DepthFrame) -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Unorm,    // Single channel — 16-bit normalized depth
            width: frame.width,
            height: frame.height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]  // Only read by shaders, never written

        let texture = device.makeTexture(descriptor: descriptor)!

        // Copy the raw uint16 bytes directly into the texture
        frame.depthPixels.withUnsafeBytes { ptr in
            texture.replace(
                region: MTLRegionMake2D(0, 0, frame.width, frame.height),
                mipmapLevel: 0,
                withBytes: ptr.baseAddress!,
                bytesPerRow: frame.width * 2   // 2 bytes per uint16
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
    //   halves every pass: 4 → 2 (default 2 passes).
    //   This fills gaps up to ~4 pixels wide — enough for genuine sensor noise
    //   gaps without smearing object silhouettes. Pass-count tuning history:
    //     8 passes, radius (voxDist+voxSize) — visibly shrank thin objects
    //     4 passes, radius voxSize           — better, still some smoothing
    //     2 passes, radius voxSize           — current. Silhouettes essentially
    //                                          untouched.
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
                     dilationSteps: Int = 2) -> MTLTexture {
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
        // maxStep = 2^dilationSteps (e.g. 4 steps → starts at 16)
        // We dispatch exactly `dilationSteps` passes — earlier this loop ran
        // `0..<maxStep` times by mistake, which executed dilationSteps useful
        // passes plus many extra no-op passes (with stepSize=0 after integer
        // division). The no-ops were harmless but the off-by-one extra 1-pixel
        // pass after the jump-flood schedule completed nibbled a pixel off
        // every silhouette every frame.
        var maxStep = 1
        for _ in 0..<dilationSteps { maxStep *= 2 }

        var stepSize = maxStep
        var texA = dilationA!
        var texB = dilationB!

        for _ in 0..<dilationSteps {
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
    //   5. Blend this new reading into the existing TSDF value with a light 0.5
    //      exponential blend (anti-shake — tuning history in VolumeIntegration.metal)
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
        var voxCount:  SIMD3<Int32>   // Full volume dimensions (1024×256×1024 in this project)
        var voxSize:   Float          // Size of each voxel in metres
        var regionMin: SIMD3<Int32>   // First voxel coordinate of the meshing region
        var regionMax: SIMD3<Int32>   // Last voxel coordinate of the meshing region
        // Buffer capacities handed to the kernels so their atomic allocators can
        // bounds-check. Must stay in sync with `struct MeshParams` in
        // SurfaceNets.metal — including the padding, which keeps
        // MemoryLayout.size equal to .stride so setBytes(length:) sends the
        // whole struct.
        var maxVerts:  Int32 = 0
        var maxTriIdx: Int32 = 0
        var _pad0:     Int32 = 0
        var _pad1:     Int32 = 0
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

    // ── coordVertMap buffer (legacy single-mesh path only) ───────────────────
    // Shared between the two Surface Nets passes:
    //   Pass 1 writes: REGION-RELATIVE cell index → vertex index (-1 if none)
    //   Pass 2 reads:  looks up the 4 surrounding vertices to form a quad
    // Region-relative indexing (see SurfaceNets.metal header) keeps this buffer
    // small: the legacy camera-box region needs ~7 MB — versus ~1 GB if it were
    // indexed by absolute volume coordinates. Grow-only and reused across calls.
    // The chunked path uses the per-slot maps in `chunkSlots` instead.
    var coordVertMapBuffer: MTLBuffer?

    // ── Pooled mesh output buffers (legacy single-mesh path only) ─────────────
    // Grow-only, reused by every meshRegion call so the legacy path doesn't
    // allocate fresh MTLBuffers per frame. The chunked path uses the
    // fixed-size buffers in `chunkSlots` instead.
    private var meshVertexBuffer: MTLBuffer?
    private var meshTriBuffer:    MTLBuffer?
    private var meshVertCountBuffer: MTLBuffer?
    private var meshTriCountBuffer:  MTLBuffer?

    // ── Chunk grid for persistent client-side caching ─────────────────────────
    // The volume is divided into fixed 32³-voxel chunks (3.2 m at 0.1 m voxels).
    // generateChunkMeshes() meshes a budgeted number of camera-near chunks per
    // frame and returns them tagged with their grid coordinate; the Quest keeps
    // a chunk dictionary and replaces only the cells that arrive — geometry
    // outside the current view persists on the client (standalone-style), and
    // the server stays stateless beyond the TSDF volume it already owns.
    let chunkSizeVox: Int32 = 32

    // Round-robin work queue of in-view chunk coords. Refilled (nearest-first)
    // whenever it runs empty, so all visible chunks refresh within a few frames
    // even though only `budget` are meshed per incoming depth frame.
    private var chunkQueue: [SIMD3<Int32>] = []

    // ── Empty-chunk backoff ───────────────────────────────────────────────────
    // Most in-range chunks are empty air (above/below the room geometry).
    // Meshing them every cycle wastes the per-frame budget and slows down how
    // often chunks with REAL geometry refresh. So chunks that come back empty
    // are re-checked with exponential backoff (skip 1, then 3, then 7 cycles,
    // capped) — new geometry appearing in a long-empty chunk is still noticed
    // within a handful of cycles, while ~80 % of the budget stays on chunks
    // that actually contain surfaces.
    private var chunkEmptyStreak: [SIMD3<Int32>: Int] = [:]
    private var chunkSkipCounter: [SIMD3<Int32>: Int] = [:]

    // Chunks we have ever sent as non-empty. Lets us send an explicit
    // "clear this cell" (vertCount == 0) exactly once when a previously
    // occupied chunk becomes empty, instead of spamming empties every cycle.
    private var nonEmptyChunks: Set<SIMD3<Int32>> = []

    // ── Pre-allocated buffer slots for BATCHED chunk meshing ─────────────────
    // All chunks of a frame are encoded into ONE command buffer with ONE
    // waitUntilCompleted. The first implementation ran a separate command
    // buffer (and GPU sync) per chunk — 8 serial round-trips a frame, which
    // tripled server-side mesh time. Each slot owns fixed-size buffers
    // matching the constant chunk-region size ((chunkSizeVox+2)³ cells).
    private struct ChunkSlot {
        let coordVertMap: MTLBuffer
        let vertexBuffer: MTLBuffer
        let triBuffer:    MTLBuffer
        let vertCount:    MTLBuffer
        let triCount:     MTLBuffer
    }
    private var chunkSlots: [ChunkSlot] = []
    private var chunkRegionCells = 0     // cells per chunk region, set when slots build
    private var chunkSlotMaxVerts = 0
    private var chunkSlotMaxTriIdx = 0

    // Build the per-slot buffer pool the first time (or grow it if budget rises).
    // Slots are fixed-size because every chunk region is the same size:
    // (chunkSizeVox + 2)³ cells (32³ chunk + 1-cell seam margin each side).
    private func ensureChunkSlots(budget: Int) {
        guard chunkSlots.count < budget else { return }

        let side = Int(chunkSizeVox) + 2
        chunkRegionCells   = side * side * side          // 39 304

        // Capacities are the MATHEMATICAL WORST CASE, not a heuristic.
        //
        // Surface Nets emits at most one vertex per cell, so a region can never
        // produce more than `chunkRegionCells` vertices. The previous
        // `chunkRegionCells / 3` was an estimate of typical density; cluttered
        // rooms exceeded it, the kernels wrote past the buffer, and the host
        // then clamped vertCount below indices that had already been written —
        // producing indices >= vertCount, which the client rejects (and which
        // crashed PhysX before that guard existed). 16.8 % of chunk responses
        // were lost to this on 2026-08-29, rising to 33.4 % on a 41-minute run.
        //
        // Each cell can emit up to 3 quads (one per axis) = 18 indices, so the
        // index buffer's true bound is cells × 18.
        //
        // Cost: ~3.9 MB per slot (157 KB map + 943 KB verts + 2.8 MB indices)
        // versus ~1.4 MB before. Trivial on an M3 Max, and it makes overflow
        // impossible by construction rather than merely unlikely.
        chunkSlotMaxVerts  = chunkRegionCells
        chunkSlotMaxTriIdx = chunkRegionCells * 18

        while chunkSlots.count < budget {
            chunkSlots.append(ChunkSlot(
                coordVertMap: device.makeBuffer(length: chunkRegionCells * 4,   options: .storageModeShared)!,
                vertexBuffer: device.makeBuffer(length: chunkSlotMaxVerts * 24, options: .storageModeShared)!,
                triBuffer:    device.makeBuffer(length: chunkSlotMaxTriIdx * 4, options: .storageModeShared)!,
                vertCount:    device.makeBuffer(length: 4, options: .storageModeShared)!,
                triCount:     device.makeBuffer(length: 4, options: .storageModeShared)!
            ))
        }
    }

    /// One meshed chunk: its grid coordinate plus the extracted triangles.
    struct ChunkMesh {
        let coord: SIMD3<Int32>
        let mesh:  MeshResult
    }

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
    //   Instead of meshing the entire 1024×256×1024 volume (~268M voxels → crash), we only mesh
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
        guard volumeTexture != nil else {
            return MeshResult(vertices: [], indices: [], vertexCount: 0, triangleCount: 0)
        }

        let voxCount = SIMD3<Int32>(frame.voxelCount.x, frame.voxelCount.y, frame.voxelCount.z)

        // ── Camera-centred bounding box ───────────────────────────────────────
        // Extract camera world position from the 4th column of the inverse view matrix
        // (viewInv transforms camera → world, so its translation column = world eye pos)
        let eyeWorld  = cameraWorldPosition(frame: frame)
        let extentVox = Int32(frame.maxUpdateDist / frame.voxelSize) + 2
        let eyeVox    = worldToVoxel(eyeWorld, voxCount: voxCount, voxelSize: frame.voxelSize)

        // Mesh only the cube around the camera, clamped to volume bounds
        let regionMin = max(eyeVox &- extentVox, SIMD3<Int32>(0, 0, 0))
        let regionMax = min(eyeVox &+ extentVox, voxCount)

        return meshRegion(regionMin: regionMin, regionMax: regionMax,
                          voxCount: voxCount, voxelSize: frame.voxelSize)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // generateChunkMeshes — Per-chunk meshing for the persistent client cache
    //
    // Returns up to `budget` chunk meshes per call. Selection:
    //   1. When the work queue is empty, refill it with every chunk whose
    //      centre lies within (maxUpdateDist + chunk radius) of the camera,
    //      sorted nearest-first.
    //   2. Pop chunks off the queue — skipping any still in empty-chunk
    //      backoff — until `budget` are selected, then encode ALL of them
    //      into one command buffer (one GPU sync per frame, see chunkSlots).
    //
    // The rotation means all in-range chunks refresh within a few frames
    // (~30 in-range chunks / 8 per frame ≈ full refresh every 4 frames
    // ≈ each chunk at ~2.5 Hz with a 10 Hz send rate) while per-frame server
    // time stays bounded.
    //
    // An empty result (vertexCount == 0, "clear this chunk" on the client) is
    // returned ONCE, when a previously non-empty chunk becomes empty; chunks
    // that stay empty are suppressed and re-checked with exponential backoff
    // (see chunkEmptyStreak / chunkSkipCounter).
    //
    // Each chunk's mesh region is expanded by 1 cell on every side. Border
    // cells get meshed by both neighbouring chunks; the duplicated seam
    // triangles are bit-identical (same TSDF inputs, same math), so they
    // render coincident with no z-fighting — this is what makes chunk
    // borders crack-free without any cross-chunk stitching.
    // ─────────────────────────────────────────────────────────────────────────
    func generateChunkMeshes(frame: DepthFrame, budget: Int) -> [ChunkMesh] {
        guard let volume = volumeTexture else { return [] }

        let voxCount = SIMD3<Int32>(frame.voxelCount.x, frame.voxelCount.y, frame.voxelCount.z)
        ensureChunkSlots(budget: budget)

        var selected: [(coord: SIMD3<Int32>, regionMin: SIMD3<Int32>, regionMax: SIMD3<Int32>)] = []
        var safety = 0
        while selected.count < budget && safety < 2048 {
            safety += 1
            if chunkQueue.isEmpty {
                chunkQueue = inRangeChunks(frame: frame, voxCount: voxCount)
                if chunkQueue.isEmpty { break }
            }
            let c = chunkQueue.removeFirst()
            if let skip = chunkSkipCounter[c], skip > 0 {
                chunkSkipCounter[c] = skip - 1
                continue
            }
            let base      = c &* chunkSizeVox
            let regionMin = max(base &- 1, SIMD3<Int32>(0, 0, 0))
            let regionMax = min(base &+ chunkSizeVox &+ 1, voxCount)
            selected.append((c, regionMin, regionMax))
        }
        guard !selected.isEmpty else { return [] }

        for i in 0..<selected.count {
            let slot = chunkSlots[i]
            memset(slot.coordVertMap.contents(), 0xFF, chunkRegionCells * 4)
            slot.vertCount.contents().storeBytes(of: Int32(0), as: Int32.self)
            slot.triCount.contents().storeBytes(of: Int32(0), as: Int32.self)
        }

        let commandBuffer = commandQueue.makeCommandBuffer()!
        for (i, sel) in selected.enumerated() {
            let slot = chunkSlots[i]
            var params = MeshParams(voxCount: voxCount, voxSize: frame.voxelSize,
                                    regionMin: sel.regionMin, regionMax: sel.regionMax,
                                    maxVerts:  Int32(chunkSlotMaxVerts),
                                    maxTriIdx: Int32(chunkSlotMaxTriIdx))
            let regionSize  = sel.regionMax &- sel.regionMin
            let regionTotal = Int(regionSize.x) * Int(regionSize.y) * Int(regionSize.z)

            let e1 = commandBuffer.makeComputeCommandEncoder()!
            e1.setComputePipelineState(vertexPipeline)
            e1.setTexture(volume, index: 0)
            e1.setBuffer(slot.vertexBuffer, offset: 0, index: 0)
            e1.setBuffer(slot.vertCount,    offset: 0, index: 1)
            e1.setBuffer(slot.coordVertMap, offset: 0, index: 2)
            e1.setBytes(&params, length: MemoryLayout<MeshParams>.size, index: 3)
            e1.dispatchThreads(
                MTLSize(width: regionTotal, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: vertexPipeline.threadExecutionWidth, height: 1, depth: 1))
            e1.endEncoding()

            let e2 = commandBuffer.makeComputeCommandEncoder()!
            e2.setComputePipelineState(indexPipeline)
            e2.setTexture(volume, index: 0)
            e2.setBuffer(slot.coordVertMap, offset: 0, index: 0)
            e2.setBuffer(slot.triBuffer,    offset: 0, index: 1)
            e2.setBuffer(slot.triCount,     offset: 0, index: 2)
            e2.setBytes(&params, length: MemoryLayout<MeshParams>.size, index: 3)
            e2.setBuffer(slot.vertexBuffer, offset: 0, index: 4)
            e2.dispatchThreads(
                MTLSize(width: regionTotal, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: indexPipeline.threadExecutionWidth, height: 1, depth: 1))
            e2.endEncoding()
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        var out: [ChunkMesh] = []
        for (i, sel) in selected.enumerated() {
            let slot = chunkSlots[i]
            let rawVerts = Int(slot.vertCount.contents().load(as: Int32.self))
            let rawTris  = Int(slot.triCount.contents().load(as: Int32.self))

            // Overflow detection. The kernels bounds-check their writes, so an
            // overflowed buffer is merely INCOMPLETE rather than corrupt — but a
            // partial chunk would still be applied by the client as if it were
            // the whole cell, leaving holes in the collision mesh. Capacities
            // are now the mathematical worst case, so this should never fire;
            // if it does, the sizing assumption is wrong and we want to know
            // rather than silently ship a broken chunk.
            if rawVerts > chunkSlotMaxVerts || rawTris > chunkSlotMaxTriIdx {
                FileHandle.standardError.write(Data(
                    "[MetalPipeline] CHUNK OVERFLOW at \(sel.coord): verts \(rawVerts)/\(chunkSlotMaxVerts), indices \(rawTris)/\(chunkSlotMaxTriIdx) — chunk dropped\n".utf8))
                continue
            }

            let vertCount = rawVerts
            let triCount  = rawTris

            if vertCount > 0 {
                chunkEmptyStreak[sel.coord] = 0
                chunkSkipCounter[sel.coord] = 0
                nonEmptyChunks.insert(sel.coord)

                let vertPtr = slot.vertexBuffer.contents().bindMemory(to: Float.self, capacity: vertCount * 6)
                let verts   = Array(UnsafeBufferPointer(start: vertPtr, count: vertCount * 6))
                let idxPtr  = slot.triBuffer.contents().bindMemory(to: UInt32.self, capacity: triCount)
                let indices = Array(UnsafeBufferPointer(start: idxPtr, count: triCount))

                out.append(ChunkMesh(coord: sel.coord,
                                     mesh: MeshResult(vertices: verts, indices: indices,
                                                      vertexCount: vertCount, triangleCount: triCount / 3)))
            } else {
                let streak = (chunkEmptyStreak[sel.coord] ?? 0) + 1
                chunkEmptyStreak[sel.coord] = streak
                chunkSkipCounter[sel.coord] = min(1 << min(streak, 3), 8) - 1

                if nonEmptyChunks.remove(sel.coord) != nil {
                    out.append(ChunkMesh(coord: sel.coord,
                                         mesh: MeshResult(vertices: [], indices: [],
                                                          vertexCount: 0, triangleCount: 0)))
                }
            }
        }
        return out
    }
    private func inRangeChunks(frame: DepthFrame, voxCount: SIMD3<Int32>) -> [SIMD3<Int32>] {
        let eyeWorld  = cameraWorldPosition(frame: frame)
        let chunkSizeWorld = Float(chunkSizeVox) * frame.voxelSize          // e.g. 3.2 m
        let chunkRadius    = chunkSizeWorld * 0.866                          // half diagonal
        let reach          = frame.maxUpdateDist + chunkRadius

        // Chunk-grid dimensions (volume voxels / chunk size, rounded up)
        let gridDims = SIMD3<Int32>(
            (voxCount.x + chunkSizeVox - 1) / chunkSizeVox,
            (voxCount.y + chunkSizeVox - 1) / chunkSizeVox,
            (voxCount.z + chunkSizeVox - 1) / chunkSizeVox
        )

        // Candidate chunk-index AABB around the camera (clamped to the grid)
        let half = SIMD3<Float>(Float(voxCount.x), Float(voxCount.y), Float(voxCount.z)) * 0.5
        func chunkIndex(_ w: Float, _ halfAxis: Float) -> Int32 {
            Int32(floor((w / frame.voxelSize + halfAxis) / Float(chunkSizeVox)))
        }
        let loX = max(chunkIndex(eyeWorld.x - reach, half.x), 0)
        let hiX = min(chunkIndex(eyeWorld.x + reach, half.x), gridDims.x - 1)
        let loY = max(chunkIndex(eyeWorld.y - reach, half.y), 0)
        let hiY = min(chunkIndex(eyeWorld.y + reach, half.y), gridDims.y - 1)
        let loZ = max(chunkIndex(eyeWorld.z - reach, half.z), 0)
        let hiZ = min(chunkIndex(eyeWorld.z + reach, half.z), gridDims.z - 1)
        guard loX <= hiX, loY <= hiY, loZ <= hiZ else { return [] }

        // Gather chunks whose centre is actually within reach (sphere test
        // tightens the AABB corner chunks away — ~half the candidate count)
        var found: [(SIMD3<Int32>, Float)] = []
        for cz in loZ...hiZ {
            for cy in loY...hiY {
                for cx in loX...hiX {
                    let coord  = SIMD3<Int32>(cx, cy, cz)
                    let centreVox = SIMD3<Float>(
                        (Float(cx) + 0.5) * Float(chunkSizeVox),
                        (Float(cy) + 0.5) * Float(chunkSizeVox),
                        (Float(cz) + 0.5) * Float(chunkSizeVox)
                    )
                    let centreWorld = (centreVox - half) * frame.voxelSize
                    let dist = simd_distance(centreWorld, eyeWorld)
                    if dist <= reach {
                        found.append((coord, dist))
                    }
                }
            }
        }
        found.sort { $0.1 < $1.1 }     // nearest first
        return found.map { $0.0 }
    }

    // Camera world position from the 4th column of the left-eye inverse view matrix.
    private func cameraWorldPosition(frame: DepthFrame) -> SIMD3<Float> {
        let eyeCol = frame.viewInv[0][3]
        return SIMD3<Float>(eyeCol.x, eyeCol.y, eyeCol.z)
    }

    // World position → voxel grid coordinate (volume centred on world origin).
    private func worldToVoxel(_ w: SIMD3<Float>, voxCount: SIMD3<Int32>, voxelSize: Float) -> SIMD3<Int32> {
        SIMD3<Int32>(
            Int32(w.x / voxelSize + Float(voxCount.x) * 0.5),
            Int32(w.y / voxelSize + Float(voxCount.y) * 0.5),
            Int32(w.z / voxelSize + Float(voxCount.z) * 0.5)
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // meshRegion — Surface Nets core for the LEGACY camera-box path
    // (generateMesh / 0x03). Extracts triangles from one axis-aligned voxel
    // region. The chunked path (generateChunkMeshes) no longer calls this —
    // it encodes its own batched dispatches using the chunkSlots buffers.
    //
    // Scratch buffers are pooled grow-only members so repeated calls don't
    // allocate. The coordVertMap is REGION-RELATIVE (matching
    // SurfaceNets.metal), so only regionTotal × 4 bytes are cleared per call
    // instead of a ~1 GB full-volume map.
    // ─────────────────────────────────────────────────────────────────────────
    private func meshRegion(
        regionMin: SIMD3<Int32>, regionMax: SIMD3<Int32>,
        voxCount: SIMD3<Int32>, voxelSize: Float
    ) -> MeshResult {
        guard let volume = volumeTexture else {
            return MeshResult(vertices: [], indices: [], vertexCount: 0, triangleCount: 0)
        }

        let regionSize  = regionMax &- regionMin
        let regionTotal = Int(regionSize.x) * Int(regionSize.y) * Int(regionSize.z)
        guard regionTotal > 0 else {
            return MeshResult(vertices: [], indices: [], vertexCount: 0, triangleCount: 0)
        }

        // Buffer size estimates based on the region:
        //   maxVertices: at most 1 vertex per 3 cells (heuristic)
        //   Sized to the mathematical worst case, matching the chunked path:
        //   Surface Nets emits at most one vertex per cell, and at most 3 quads
        //   (18 indices) per cell. The former `regionTotal / 3` and `× 6` were
        //   density estimates that cluttered rooms exceeded, which produced
        //   indices past the clamped vertex count. See ensureChunkSlots().
        let maxVertices   = max(regionTotal, 1)
        let maxTriIndices = regionTotal * 18

        // ── (Re)size pooled buffers, grow-only ────────────────────────────────
        if coordVertMapBuffer == nil || coordVertMapBuffer!.length < regionTotal * 4 {
            coordVertMapBuffer = device.makeBuffer(length: regionTotal * 4, options: .storageModeShared)!
        }
        if meshVertexBuffer == nil || meshVertexBuffer!.length < maxVertices * 24 {
            meshVertexBuffer = device.makeBuffer(length: maxVertices * 24, options: .storageModeShared)!
        }
        if meshTriBuffer == nil || meshTriBuffer!.length < maxTriIndices * 4 {
            meshTriBuffer = device.makeBuffer(length: maxTriIndices * 4, options: .storageModeShared)!
        }
        if meshVertCountBuffer == nil {
            meshVertCountBuffer = device.makeBuffer(length: 4, options: .storageModeShared)!
            meshTriCountBuffer  = device.makeBuffer(length: 4, options: .storageModeShared)!
        }

        // Clear the region's map cells to INVALID_VERTEX (-1 = 0xFFFFFFFF)
        memset(coordVertMapBuffer!.contents(), 0xFF, regionTotal * 4)

        // Zero the atomic counters
        meshVertCountBuffer!.contents().storeBytes(of: Int32(0), as: Int32.self)
        meshTriCountBuffer!.contents().storeBytes(of: Int32(0), as: Int32.self)

        var params = MeshParams(
            voxCount:  voxCount,
            voxSize:   voxelSize,
            regionMin: regionMin,
            regionMax: regionMax,
            maxVerts:  Int32(maxVertices),
            maxTriIdx: Int32(maxTriIndices)
        )

        let commandBuffer = commandQueue.makeCommandBuffer()!

        // ── Pass 1: Generate vertices ─────────────────────────────────────────
        let vertEncoder = commandBuffer.makeComputeCommandEncoder()!
        vertEncoder.setComputePipelineState(vertexPipeline)
        vertEncoder.setTexture(volume,                  index: 0)
        vertEncoder.setBuffer(meshVertexBuffer!,        offset: 0, index: 0)
        vertEncoder.setBuffer(meshVertCountBuffer!,     offset: 0, index: 1)
        vertEncoder.setBuffer(coordVertMapBuffer!,      offset: 0, index: 2)
        vertEncoder.setBytes(&params, length: MemoryLayout<MeshParams>.size, index: 3)

        let w1 = vertexPipeline.threadExecutionWidth
        vertEncoder.dispatchThreads(
            MTLSize(width: regionTotal, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w1, height: 1, depth: 1)
        )
        vertEncoder.endEncoding()

        // ── Pass 2: Generate triangle indices ─────────────────────────────────
        let idxEncoder = commandBuffer.makeComputeCommandEncoder()!
        idxEncoder.setComputePipelineState(indexPipeline)
        idxEncoder.setTexture(volume,               index: 0)
        idxEncoder.setBuffer(coordVertMapBuffer!,   offset: 0, index: 0)
        idxEncoder.setBuffer(meshTriBuffer!,        offset: 0, index: 1)
        idxEncoder.setBuffer(meshTriCountBuffer!,   offset: 0, index: 2)
        idxEncoder.setBytes(&params, length: MemoryLayout<MeshParams>.size, index: 3)
        idxEncoder.setBuffer(meshVertexBuffer!,     offset: 0, index: 4)

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
        let vertCount = min(Int(meshVertCountBuffer!.contents().load(as: Int32.self)), maxVertices)
        let triCount  = min(Int(meshTriCountBuffer!.contents().load(as: Int32.self)), maxTriIndices)

        let vertPtr = meshVertexBuffer!.contents().bindMemory(to: Float.self, capacity: vertCount * 6)
        let verts   = Array(UnsafeBufferPointer(start: vertPtr, count: vertCount * 6))

        let idxPtr  = meshTriBuffer!.contents().bindMemory(to: UInt32.self, capacity: triCount)
        let indices = Array(UnsafeBufferPointer(start: idxPtr, count: triCount))

        return MeshResult(
            vertices:      verts,
            indices:       indices,
            vertexCount:   vertCount,
            triangleCount: triCount / 3
        )
    }
}
