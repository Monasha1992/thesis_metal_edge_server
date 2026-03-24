# EdgeMetalServer

Native Swift + Metal edge-compute server for the **Mixed-Reality Laser Tag** thesis project.

Receives stereo depth frames from a Meta Quest 3 over WiFi, runs the full TSDF pipeline on the M3 Max GPU using Metal Shading Language, generates surface-net meshes on the CPU, and streams them back to the headset.

---

## Requirements

| Requirement | Minimum |
|---|---|
| Mac | M-series (Apple Silicon) — Metal GPU required |
| macOS | 14 Sonoma or later (`Metal read_write` 3D texture support) |
| Xcode Command Line Tools | 15 or later (Swift 5.9+) |
| Network | Same WiFi network as the Quest 3 (WiFi 6 recommended) |

Check your Swift version:
```bash
swift --version
```

Install Xcode Command Line Tools if missing:
```bash
xcode-select --install
```

---

## Project structure

```
EdgeMetalServer/
├── Package.swift
├── .gitignore
├── README.md
└── Sources/EdgeMetalServer/
    ├── main.swift           # Entry point — CLI args, startup, per-client loop
    ├── Log.swift            # Levelled logger (debug/info/warn/error + timestamps)
    ├── TCPServer.swift      # POSIX TCP accept loop (port 9555 by default)
    ├── Protocol.swift       # Wire format matching EdgeProtocol.cs on the Quest
    ├── DepthProcessor.swift # Metal pipeline — DepthNorm → Dilation → TSDF Integrate
    ├── Shaders.swift        # MSL compute shaders compiled at runtime
    ├── VoxelVolume.swift    # Chunk-grid layout over the 3-D voxel volume
    ├── MeshGenerator.swift  # Surface-nets mesher (port of NetMesher.cs)
    └── MetricsLogger.swift  # Per-frame CSV performance logger
```

---

## Build

```bash
cd /path/to/EdgeMetalServer

# Debug build (slower GPU, full symbols — good for development)
swift build

# Release build (full optimisation — use for thesis measurements)
swift build -c release
```

Build output is placed in `.build/debug/EdgeMetalServer` or `.build/release/EdgeMetalServer`.

---

## Run

### Quickstart (defaults)

```bash
swift run -c release EdgeMetalServer
```

Listens on **port 9555**, creates `edge_metrics.csv` in the working directory, uses a 128³ voxel volume at 0.1 m resolution.

### All CLI arguments

```bash
swift run -c release EdgeMetalServer \
  --port=9555 \
  --loglevel=info \
  --metrics=edge_metrics.csv \
  --voxcount=128 \
  --voxsize=0.1 \
  --maxdist=6.0 \
  --chunksize=5.0
```

| Argument | Default | Description |
|---|---|---|
| `--port` | `9555` | TCP port the Quest connects to |
| `--loglevel` | `info` | `debug` / `info` / `warn` / `error` |
| `--metrics` | `edge_metrics.csv` | Path for the per-frame metrics CSV |
| `--voxcount` | `128` | Voxel grid side length (NxNxN) |
| `--voxsize` | `0.1` | Voxel size in metres |
| `--maxdist` | `6.0` | Maximum TSDF integration distance in metres |
| `--chunksize` | `5.0` | World-space side length of each mesh chunk in metres |

### Log levels

| Level | What you see |
|---|---|
| `error` | Fatal errors only (stderr) |
| `warn` | Recoverable anomalies — bad packets, empty meshes |
| `info` | Startup config, connection events, every-10-frame summary **(recommended for experiments)** |
| `debug` | Every frame, every GPU stage timing, every chunk readback and mesh output |

---

## Connecting the Quest

1. Ensure the Mac and the Quest are on the **same WiFi network**.
2. Find the Mac's local IP address:
   ```bash
   ipconfig getifaddr en0
   ```
3. In the Unity project on the Quest, set `EdgeServerSettings` → `Server IP` to that address and `Port` to `9555`.
4. Start the server first, then launch the Unity scene on the Quest.

The server accepts **one client at a time**. If the Quest disconnects it reconnects automatically; the server keeps listening.

---

## GPU pipeline

Each depth frame received triggers this pipeline, fully on the M3 Max GPU:

```
Quest depth frame (R16Unorm stereo)
        │
        ▼
[1] DepthNorm        — reconstruct world-space normals from raw depth
        │
        ▼
[2] InitDepthDilation — copy depth slice 0 into ping-pong texture
        │
        ▼
[3] DilateDepthStep × 8 — jump-flood dilation to fill depth gaps
        │
        ▼
[4] Integrate        — TSDF voxel integration into 3-D R32Float volume
        │
        ▼
[5] Readback (CPU)   — blit 3-D texture region → staging buffer per chunk
        │
        ▼
[6] Surface nets     — dual-contouring mesher produces vertices + normals + indices
        │
        ▼
Quest ← MeshChunk packets (TCP)
```

All compute shaders are written in **Metal Shading Language (MSL)** and compiled at runtime from the string constant in `Shaders.swift` — no `.metal` files or Xcode build rules required.

---

## Metrics CSV

Every processed depth frame appends one row to the CSV:

| Column | Unit | Description |
|---|---|---|
| `timestampMs` | ms | Unix timestamp of the depth frame from the Quest |
| `depthNormMs` | ms | GPU time: kernelDepthNorm |
| `dilationMs` | ms | GPU time: depth dilation (all steps) |
| `integrateMs` | ms | GPU time: kernelIntegrate (TSDF) |
| `meshMs` | ms | CPU time: readback + surface-nets meshing (all chunks) |
| `sendMs` | ms | TCP send time for all mesh chunk packets |
| `roundTripMs` | ms | Quest frame timestamp → server sends first chunk |
| `bytesSent` | bytes | Total bytes sent for this frame (including framing) |
| `chunksGenerated` | — | Number of non-empty mesh chunks sent |
| `serverQueueDepth` | — | Incoming frame queue depth at dispatch time |

---

## Network throttling (thesis experiments)

Use macOS `dnctl` + `pfctl` to simulate degraded WiFi conditions on the Mac side without touching router settings (required because QoS disables NAT acceleration on the RT-AX55).

### Add throttle on port 9555

```bash
# Create a pipe: 10 Mbit/s, 20ms extra delay, 1% packet loss
sudo dnctl pipe 1 config bw 10Mbit/s delay 20 plr 0.01

# Anchor rule: apply pipe 1 to TCP traffic on port 9555 (both directions)
echo "dummynet in  proto tcp from any to any port 9555 pipe 1" | sudo pfctl -f -
echo "dummynet out proto tcp from any to any port 9555 pipe 1" | sudo pfctl -f -
sudo pfctl -e
```

### Remove throttle

```bash
sudo pfctl -d
sudo dnctl pipe 1 delete
```

### Suggested test matrix

| Scenario | Bandwidth | Extra delay | Packet loss |
|---|---|---|---|
| Baseline (no throttle) | — | 0 ms | 0% |
| Mild congestion | 50 Mbit/s | 5 ms | 0% |
| Moderate congestion | 20 Mbit/s | 20 ms | 0.5% |
| Heavy congestion | 5 Mbit/s | 50 ms | 1% |
| Near-failure | 2 Mbit/s | 100 ms | 5% |

---

## Profiling on M3 Max

### GPU timing from the server

Run with `--loglevel=debug` to see per-stage GPU timings every frame:

```
[ 12.345s][DBG][gpu] GPU done  norm=0.41ms  dilation=1.23ms  integrate=4.87ms  total=6.51ms
```

### powermetrics (system-wide GPU/CPU/power)

```bash
sudo powermetrics \
  --samplers gpu_power,cpu_power,thermal \
  --sample-rate 1000 \
  -o powermetrics_edge.txt
```

Run this alongside the server during each experiment. Correlate timestamps with the metrics CSV.

### Metal GPU timeline (Instruments)

```bash
# Open Instruments and profile the running server process
instruments -t "Metal System Trace" -p $(pgrep EdgeMetalServer)
```

Or open Instruments.app → File → Attach to Process → EdgeMetalServer → Metal System Trace template.

---

## Troubleshooting

**`No Metal device found`**
- Must run on Apple Silicon. Intel Macs do not support the required `read_write` access on 3-D textures in Metal.

**`bind() failed errno=48`**
- Port 9555 is already in use. Kill the old process: `lsof -ti:9555 | xargs kill`

**Quest connects but no meshes appear**
- Check `--loglevel=debug` output for `Chunk (...): no surface` — the TSDF volume may need a few frames to accumulate surface data before crossings appear.
- Verify the Quest's `Server IP` matches `ipconfig getifaddr en0`.

**Very high `roundTripMs`**
- Check WiFi band — ensure the Quest is on 5 GHz, not 2.4 GHz.
- Check for other heavy traffic on the network during experiments.

**MSL compile error on startup**
- The full error is printed to stderr. This indicates a shader bug introduced during development. The compile happens once at startup from `Shaders.swift`.

---

## Wire protocol

The protocol is defined in `Protocol.swift` (Swift) and must stay in sync with `Assets/Anaglyph/EdgeServer/EdgeProtocol.cs` (Unity/C#).

```
Framing:   [4 bytes LE int32 length] [N bytes payload]
Payload:   byte 0 = PacketType (0x01 = DepthFrame, 0x02 = MeshChunk)
```

**DepthFrame** (Quest → Mac): timestamp, width, height, 2× projection matrix, 2× view matrix, near, far, R16Unorm pixel data for left and right eye slices.

**MeshChunk** (Mac → Quest): depth timestamp, server timing fields, world position, vertex count, index count, vertices (float3[]), normals (float3[]), indices (int32[]), queue depth.

All matrices are written **row-major** from C# and transposed on read in Swift to produce column-major `simd_float4x4`.
