# EdgeMetalServer

The Mac-side edge server for the AR laser-tag thesis. It receives raw depth
frames from a Quest 3 over Wi-Fi TCP, runs the full GPU reconstruction pipeline
on the Mac (Metal: depth dilation → normals → TSDF integration → Surface Nets
meshing), and streams the resulting triangle mesh back. Each instance serves
**one** headset; run several instances on different ports to serve several
headsets.

- Requires macOS 14+ and Swift 6.3+ (Apple Silicon recommended — tested on M3 Max).
- Wire protocol: see [`../docs/04-protocol.md`](../docs/04-protocol.md).
- Full build/deploy guide: [`../docs/07-build-deploy.md`](../docs/07-build-deploy.md).

## Build once, run the binary (recommended)

Build the optimized binary a single time, then launch it directly — one process
per headset, each on its own port:

```bash
cd EdgeMetalServer
swift build -c release                       # build once (cached afterwards)

.build/release/EdgeMetalServer               # headset 1 → port 9900 (default)
.build/release/EdgeMetalServer 9901          # headset 2 → port 9901 (new terminal)
.build/release/EdgeMetalServer 9902          # headset 3 → port 9902 (new terminal)
```

Each instance prints `Server listening on port <N>`. Stop one with `Ctrl-C`.
Rebuild (`swift build -c release`) only after you change `.swift` or `.metal`
source.

> **Always use the release build for data collection.** The debug build adds
> real overhead that inflates the `server_*_ms` timings recorded in the CSVs.

### Why `.build/release/EdgeMetalServer <port>` and not `swift run ... <port>`

`swift run` treats the first word after it as the *product name*, so
`swift run -c release 9901` fails with "no executable product named '9901'".
The build-once binary takes the port directly as its first argument, with no
such ambiguity. (If you do use `swift run`, the correct form is
`swift run -c release EdgeMetalServer 9901`.)

## Ports & the one-client-per-instance model

- Base port is **9900**; the port is the binary's first CLI argument (default
  9900 if omitted).
- Each instance serves a single headset. A second headset connecting to a busy
  instance is refused, so the headset's client scans upward (9900 → 9901 → …)
  until it finds a free instance — no per-device configuration needed (same APK
  on every headset).
- On connect, a free instance greets the client with one `0xAA` byte; a busy one
  closes the connection without greeting. See the handshake section in
  [`../docs/04-protocol.md`](../docs/04-protocol.md).

## Output logs

Each instance writes a per-frame CSV to:

```
~/EdgeMetalServer/logs/server_p<port>_<UTC-timestamp>.csv
```

The `p<port>` in the filename identifies which instance/headset it came from.
Join it to the matching Quest CSV (`metrics_port<port>_*.csv`) on
`timestamp_echo_ms` = `server_ts_ms`.

## Running in Xcode (debugging only)

Open `Package.swift` in Xcode, then **Product → Scheme → Edit Scheme… → Run →
Arguments** and add the port (e.g. `9901`) under *Arguments Passed On Launch*.
For multiple concurrent instances, duplicate the scheme per port. Use Xcode only
to step through server code — collect measurements with the release binary above.

## Layout

```
Sources/EdgeMetalServer/
├── EdgeMetalServer.swift   # entry point: TCP listener, busy-guard, framing, send
├── DepthFrame.swift        # wire-format parser (depth packet → struct)
├── MetalPipeline.swift     # GPU pipeline driver (textures, buffers, dispatches)
├── MetricsRecorder.swift   # per-frame CSV writer
└── Shaders/                # Metal compute kernels (compiled into Bundle.module)
    ├── DepthDilation.metal
    ├── DepthNormal.metal
    ├── DepthProcess.metal
    ├── SurfaceNets.metal
    └── VolumeIntegration.metal
```
