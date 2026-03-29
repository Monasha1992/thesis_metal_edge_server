import Foundation
import simd

struct DepthFrame {
    let view: [float4x4]  // 2 eyes
    let proj: [float4x4]  // 2 eyes
    let viewInv: [float4x4]  // 2 eyes
    let projInv: [float4x4]  // 2 eyes
    
    let voxelCount: SIMD3<Int32>
    let voxelSize: Float
    let voxelDist: Float
    let maxUpdateDist: Float
    let numPlayers: Int32
        
    let depthPixels: Data  // raw float32 pixels
    let width: Int
    let height: Int
}

func parseDepthFrame(_ data: Data) -> DepthFrame {
    var offset = 0

    func readFloat() -> Float {
        var val: Float = 0
        _ = withUnsafeMutableBytes(of: &val) { data.copyBytes(to: $0, from: offset..<offset+4) }
        offset += 4
        return val
    }

    func readInt32() -> Int32 {
        var val: Int32 = 0
        _ = withUnsafeMutableBytes(of: &val) { data.copyBytes(to: $0, from: offset..<offset+4) }
        offset += 4
        return val
    }

    func readMatrix() -> float4x4 {
        var cols = [SIMD4<Float>](repeating: .zero, count: 4)
        for col in 0..<4 {
            let x = readFloat()
            let y = readFloat()
            let z = readFloat()
            let w = readFloat()
            cols[col] = SIMD4<Float>(x, y, z, w)
        }
        return float4x4(cols)
    }

    let view = [readMatrix(), readMatrix()]
    let proj = [readMatrix(), readMatrix()]
    let viewInv = [readMatrix(), readMatrix()]
    let projInv = [readMatrix(), readMatrix()]

    let voxelCount = SIMD3<Int32>(readInt32(), readInt32(), readInt32())
    let voxelSize = readFloat()
    let voxelDist = readFloat()
    let maxUpdateDist = readFloat()
    let numPlayers = readInt32()

    let depthPixels = data.subdata(in: offset..<data.count)
    let pixelCount = depthPixels.count / 4
    let width = 320
    let height = pixelCount / width

    return DepthFrame(
        view: view, proj: proj,
        viewInv: viewInv, projInv: projInv,
        voxelCount: voxelCount,
        voxelSize: voxelSize,
        voxelDist: voxelDist,
        maxUpdateDist: maxUpdateDist,
        numPlayers: numPlayers,
        depthPixels: depthPixels,
        width: width, height: height
    )
}
