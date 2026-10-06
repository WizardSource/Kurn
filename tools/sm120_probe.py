# ruff: noqa: E501  (one PTX instruction per probe string)
"""Which PTX instructions ptxas accepts per target (no GPU needed): the sm_120 (RTX 50) feature probe.

    python tools/sm120_probe.py [--archs sm_80,sm_120,sm_120a,sm_120f] [--ptxas PATH]

Each probe is a minimal PTX kernel using one instruction; the table says whether ptxas assembles it for each target.
"""

import argparse
import os
import shutil
import subprocess
import tempfile

REGS = """
  .reg .b32 a<4>, b<2>, sa, sb, h, x, y;
  .reg .f32 c<4>, d<4>;
  .reg .b64 p, tm, mb, sm;
  .reg .b16 q, mask;
  .shared .align 128 .b8 buf[1024];
  .shared .align 8 .b64 bar;
  mov.u32 a0, 0; mov.u32 a1, 0; mov.u32 a2, 0; mov.u32 a3, 0; mov.u32 b0, 0; mov.u32 b1, 0;
  mov.u32 sa, 0x7f7f7f7f; mov.u32 sb, 0x7f7f7f7f; mov.u16 mask, 3;
  mov.f32 c0, 0f00000000; mov.f32 c1, 0f00000000; mov.f32 c2, 0f00000000; mov.f32 c3, 0f00000000;
  ld.param.u64 p, [P];
  mov.u64 tm, p;
  mov.u64 mb, bar;
  mov.u64 sm, buf;
"""
STORE = "  st.global.f32 [p], d0; st.global.f32 [p+4], d1; st.global.f32 [p+8], d2; st.global.f32 [p+12], d3;\n"
MMA_OPS = "{d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3}"

PROBES = {
    "mma f16 m16n8k16 (kurn engine, attention)": f"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {MMA_OPS};",
    "mma bf16 m16n8k16": f"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {MMA_OPS};",
    "cp.async 16B + ldmatrix (sm_80 pipeline)": "cp.async.cg.shared.global [buf], [p], 16; cp.async.wait_all;"
    " ldmatrix.sync.aligned.m8n8.x4.shared.b16 {a0, a1, a2, a3}, [buf];",
    "mma FP8 e4m3, f32 accumulate (m16n8k32)": f"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {MMA_OPS};",
    "mma FP8 e4m3, f16 accumulate (m16n8k32)": "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {a0, a1}, {a0, a1, a2, a3}, {b0, b1}, {b0, b1};",
    "mma kind::f8f6f4 FP4 e2m1 (no scales)": f"mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 {MMA_OPS};",
    "mma kind::mxf8f6f4 block_scale (MXFP8, ue8m0)": "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0"
    f" {MMA_OPS}, sa, {{0, 1}}, sb, {{0, 1}};",
    "mma kind::mxf4 block_scale (MXFP4, ue8m0 per 32)": "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0"
    f" {MMA_OPS}, sa, {{0, 1}}, sb, {{0, 1}};",
    "mma kind::mxf4nvf4 block_scale 4X (NVFP4, ue4m3 per 16)": "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32"
    f".ue4m3 {MMA_OPS}, sa, {{0, 0}}, sb, {{0, 0}};",
    "cvt f32x2 -> e4m3x2 (FP8 KV write)": "cvt.rn.satfinite.e4m3x2.f32 q, c1, c0;",
    "cvt f32x2 -> e2m1x2 (FP4 quantize)": "{ .reg .b8 r; cvt.rn.satfinite.e2m1x2.f32 r, c1, c0; }",
    "TMA cp.async.bulk.tensor 2d": "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [buf], [tm, {a0, a1}], [bar];",
    "TMA multicast::cluster": "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
    " [buf], [tm, {a0, a1}], [bar], mask;",
    "cluster barrier (thread block clusters)": "barrier.cluster.arrive; barrier.cluster.wait;",
    "setmaxnreg (warp-specialized register split)": "setmaxnreg.inc.sync.aligned.u32 192;",
    "wgmma (Hopper warpgroup MMA)": "wgmma.fence.sync.aligned;",
    "tcgen05 / TMEM (datacenter Blackwell)": "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [buf], 32;",
}
NEEDS_MMA_STORE = ("mma ",)


def ptx(arch, body):
    store = STORE if body.startswith("mma") and "f16 {a0" not in body else ""
    return (f".version 8.8\n.target {arch}\n.address_size 64\n.visible .entry probe(.param .u64 P)\n{{\n{REGS}  {body}\n{store}"
            "  ret;\n}\n")  # fmt: skip


def run(ptxas, arch, body):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.ptx")
        with open(src, "w") as fh:
            fh.write(ptx(arch, body))
        r = subprocess.run([ptxas, f"-arch={arch}", src, "-o", os.path.join(d, "p.cubin")], capture_output=True, text=True)
    if r.returncode == 0:
        return "yes", ""
    msg = next((ln for ln in r.stderr.splitlines() if "error" in ln), r.stderr.strip())
    return "no", msg.split("error   :")[-1].strip()[:140]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="sm_80,sm_90a,sm_100a,sm_120,sm_120a,sm_120f")
    ap.add_argument("--ptxas", default=shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas")
    ap.add_argument("--why", action="store_true", help="print ptxas's reason for each rejection")
    a = ap.parse_args()
    archs = a.archs.split(",")
    ver = subprocess.run([a.ptxas, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    print(f"ptxas: {ver}\n")
    print("| instruction | " + " | ".join(archs) + " |")
    print("|---|" + "---|" * len(archs))
    reasons = []
    for name, body in PROBES.items():
        cells = []
        for ar in archs:
            ok, why = run(a.ptxas, ar, body)
            cells.append(ok)
            if why:
                reasons.append(f"{name} / {ar}: {why}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    if a.why:
        print()
        for r in reasons:
            print("-", r)


if __name__ == "__main__":
    main()
