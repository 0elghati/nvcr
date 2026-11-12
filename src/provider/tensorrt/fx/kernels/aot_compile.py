#!/usr/bin/env python3
"""Compiles the nvcr.fx.v1 Triton kernels ahead of time into a C++ table of CUDA binaries.

usage: aot_compile.py --variants configs/tensorrt/fx-kernel-variants.json --out FILE.cpp --header kernel_table.hpp
       aot_compile.py --programs DIR [DIR ...] --write-variants configs/tensorrt/fx-kernel-variants.json

The kernel variants come from fx programs (every gemm node's group, residual count, activation, input shift and
divisibility of its sizes; both divisibilities of the run-time sizes); the second form records them so builds do
not need the programs. Like Triton's JIT, an integer argument gets a divisibility-by-16 hint exactly when its value is
divisible by 16; the plugin picks the matching variant at run time. Hints and tiles change speed only: the kernels
use integer arithmetic. Requires Triton (tested: 3.5.1).
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import triton
import triton.backends

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fx_kernels as K  # noqa: E402

RCAP = 2 ** 29 - 1
D2 = 32639
OFF = 128 * (256 ** 2 - 1) // 255
TILE = dict(BM=64, BN=128, BK=32, warps=4, stages=3)  # keep in sync with plans.cpp


def gemm_key(v):
    return ('gemm_g{G}_r{NRES}_ws{WSILU}_wsh{WSH}_xs{XS}_bm{BM}_bn{BN}_bk{BK}_w{warps}_s{stages}'
            '_ho{hO}_hk{hK}_hs{hS}_hl{hL}_hn{hN}').format(**v)


def dw_key(v):
    return 'dw_th4_bw128_w4_hh{hH}_hw{hW}'.format(**v)


def variants(directories):
    gemms, dws = {}, {}
    for directory in directories:
        for path in sorted(Path(directory).glob('*.fxprog.json')):
            program = json.loads(path.read_text())
            tables = program.get('delta_tables', [])
            for node in program['nodes']:
                if node['op'] == 'depthwise3x3':
                    for hh in (0, 1):
                        for hw in (0, 1):
                            v = dict(hH=hh, hW=hw)
                            dws[dw_key(v)] = v
                if node['op'] != 'gemm':
                    continue
                o, k = node['out_channels'], node['in_channels']
                ws = node['delta_table'] >= 0
                h = tables[node['delta_table']]['h'] if ws else 0
                for hn in (0, 1):
                    v = dict(TILE, G=node['group'], NRES=len(node['inputs']) - 1, WSILU=int(ws),
                             WSH=tables[node['delta_table']]['shift'] if ws else 0, XS=node['input_shift'],
                             hO=int(o % 16 == 0), hK=int(k % 16 == 0), hS=int(o * k % 16 == 0),
                             hL=int(ws and h % 16 == 0 and h != 0), hN=hn)
                    gemms[gemm_key(v)] = v
    return gemms, dws


def compile_one(fn, signature, constants, hints, num_warps, num_stages, target):
    sig = dict(signature)
    for name in constants:
        sig[name] = 'constexpr'
    idx = {name: i for i, name in enumerate(fn.arg_names)}
    attrs = {(idx[a],): [['tt.divisibility', 16]] for a in hints}
    src = triton.compiler.ASTSource(fn=fn, constexprs=constants, signature=sig, attrs=attrs)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({'num_warps': num_warps, 'num_stages': num_stages})
    cc = triton.compile(src, target=target, options=options.__dict__)
    if getattr(cc.metadata, 'global_scratch_size', 0) > 0 or getattr(cc.metadata, 'profile_scratch_size', 0) > 0:
        raise RuntimeError(f'{fn.__name__}: kernels with scratch memory are not supported')
    return cc.asm['cubin'], cc.metadata.name, cc.metadata.shared


def gemm(v, target):
    ptrs = {'w_ptr': '*i8', 'x_ptr': '*i32', 'clo_ptr': '*i32', 'chi_ptr': '*i32', 's_ptr': '*i32',
            'lut_ptr': '*i16' if v['WSILU'] else '*i32', 'r1_ptr': '*i32', 'r2_ptr': '*i32', 'out_ptr': '*i32',
            'amax_ptr': '*i32'}
    ints = {'O': 'i32', 'K': 'i32', 'N': 'i32', 'stride_wd': 'i32', 'NLUT': 'i32'}
    const = {'LW': 2, 'LA': 2, 'OFF': OFF, 'WSILU': bool(v['WSILU']), 'G': v['G'], 'NRES': v['NRES'], 'RC': RCAP,
             'XS': v['XS'], 'XCAP': D2, 'CNT': False, 'WSH': v['WSH'], 'BM': v['BM'], 'BN': v['BN'], 'BK': v['BK']}
    hints = list(ptrs) + [name for name, flag in (('O', 'hO'), ('K', 'hK'), ('stride_wd', 'hS'), ('NLUT', 'hL'),
                                                  ('N', 'hN')) if v[flag]]
    return compile_one(K._exact_gemm2_kernel, {**ptrs, **ints}, const, hints, v['warps'], v['stages'], target)


def dw(v, target):
    ptrs = {'x_ptr': '*i32', 'w_ptr': '*i64', 'b_ptr': '*i64', 's_ptr': '*i64', 'out_ptr': '*i32', 'amax_ptr': '*i32'}
    hints = list(ptrs) + (['H'] if v['hH'] else []) + (['W'] if v['hW'] else [])
    return compile_one(K._dw3x3_kernel, {**ptrs, 'H': 'i32', 'W': 'i32'}, {'TH': 4, 'BW': 128}, hints, 4, 3, target)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--programs', nargs='*', default=[])
    p.add_argument('--write-variants')
    p.add_argument('--variants')
    p.add_argument('--out')
    p.add_argument('--header')
    p.add_argument('--target', default=None, help='backend:arch:warp, e.g. cuda:89:32 (default: this GPU)')
    a = p.parse_args()
    if a.write_variants:
        gemms, dws = variants(a.programs)
        rows = {'gemm': [json.dumps(v) for v in sorted(gemms.values(), key=gemm_key)],
                'dw': [json.dumps(v) for v in sorted(dws.values(), key=dw_key)]}
        Path(a.write_variants).write_text('{\n' + ',\n'.join(
            f'"{kind}": [\n  ' + ',\n  '.join(items) + '\n]' for kind, items in rows.items()) + '\n}\n')
        return
    target = (triton.backends.compiler.GPUTarget(*[int(x) if x.isdigit() else x for x in a.target.split(':')])
              if a.target else triton.runtime.driver.active.get_current_target())
    spec = json.loads(Path(a.variants).read_text()) if a.variants else {}
    gemms = {gemm_key(v): v for v in spec.get('gemm', [])}
    dws = {dw_key(v): v for v in spec.get('dw', [])}
    entries, blobs = [], []
    for table, compile_fn in ((gemms, gemm), (dws, dw)):
        for key, v in sorted(table.items()):
            cubin, name, shared = compile_fn(v, target)
            sym = 'cubin_' + hashlib.sha256(key.encode()).hexdigest()[:16]
            blobs.append(f'alignas(16) const unsigned char {sym}[{len(cubin)}] = {{{",".join(map(str, cubin))}}};')
            entries.append(f'    {{"{key}", {sym}, {len(cubin)}, "{name}", {shared}, {v.get("warps", 4)}}},')
    if not entries:  # a build without programs or Triton: an empty table
        entries.append('    {"", nullptr, 0, "", 0, 0},')
    text = ['// Generated by src/provider/tensorrt/fx/kernels/aot_compile.py: do not edit.',
            f'// Triton {triton.__version__}, target {target.backend}:{target.arch}:{target.warp_size}.',
            f'#include "{a.header}"', '', 'namespace nvcr::tensorrt::fx {', 'namespace {'] + blobs + [
            '}  // namespace', '', 'const KernelEntry kernel_entries[] = {'] + entries + [
            '};',
            f'const std::size_t kernel_entry_count = {len(gemms) + len(dws)};',
            f'const char kernel_target[] = "{target.backend}:{target.arch}";',
            f'const char kernel_triton_version[] = "{triton.__version__}";',
            '}  // namespace nvcr::tensorrt::fx', '']
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_text('\n'.join(text))
    os.replace(tmp, out)
    print(f'{len(gemms) + len(dws)} kernels -> {out}')


if __name__ == '__main__':
    main()
