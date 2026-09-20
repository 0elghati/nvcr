#!/usr/bin/env python3
"""Versioned offline metrics. No imports from either codec implementation."""
from __future__ import annotations
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path

QUALITY_CONTRACT = 'decoded-yuv420p8-pooled-plane-6-1-1-v1'
TIMING_CONTRACT = 'host-yuv420p8-completed-frame-v1'


def digest(path: Path, count: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        remaining = count
        while remaining is None or remaining > 0:
            data = f.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not data:
                if remaining: raise ValueError(f'truncated input: {path}')
                break
            h.update(data)
            if remaining is not None: remaining -= len(data)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def psnr(sse: int, count: int):
    # JSON has no infinity number. Preserve exact reconstruction explicitly.
    return 'infinity' if sse == 0 else 10 * math.log10(255 ** 2 * count / sse)


def weighted(values):
    if 'infinity' in values: return 'infinity'
    return (6 * values[0] + values[1] + values[2]) / 8


def quality(reference: Path, decoded: Path, width: int, height: int, frames: int) -> dict:
    import numpy as np
    if min(width, height, frames) <= 0 or width % 2 or height % 2:
        raise ValueError('positive even YUV420P8 dimensions and positive frame count required')
    sizes = [width * height, width * height // 4, width * height // 4]
    frame_bytes = sum(sizes)
    if reference.stat().st_size % frame_bytes or reference.stat().st_size < frames * frame_bytes:
        raise ValueError('reference frame alignment/count mismatch')
    if decoded.stat().st_size != frames * frame_bytes:
        raise ValueError('decoded dimensions/frame count mismatch or truncated/trailing output')
    rows = []
    with reference.open('rb') as a, decoded.open('rb') as b:
        for index in range(frames):
            planes = []
            for label, size in zip('YUV', sizes):
                x = np.frombuffer(a.read(size), dtype=np.uint8).astype(np.int32)
                y = np.frombuffer(b.read(size), dtype=np.uint8).astype(np.int32)
                error = x - y
                sse = int(np.sum(error * error, dtype=np.int64))
                planes.append({'plane': label, 'samples': size, 'sse': sse,
                               'psnr_db': psnr(sse, size)})
            rows.append({'frame_index': index, 'planes': planes,
                         'psnr_yuv_db': weighted([p['psnr_db'] for p in planes])})
    pooled = [psnr(sum(r['planes'][p]['sse'] for r in rows), sizes[p] * frames) for p in range(3)]
    means = []
    for p in range(3):
        values = [r['planes'][p]['psnr_db'] for r in rows]
        means.append('infinity' if 'infinity' in values else statistics.mean(values))
    return {'schema': 'nvcr.measurement.quality.v1', 'contract': QUALITY_CONTRACT,
            'domain': 'serialized decoded planar Y,U,V uint8; full code range 0..255',
            'evaluator_conversion': 'none; no resizing, clipping, rounding or colour conversion',
            'width': width, 'height': height, 'frames': frames,
            'reference_evaluated_sha256': digest(reference, frame_bytes * frames),
            'decoded_sha256': digest(decoded), 'per_frame': rows,
            'pooled_plane_psnr_db': dict(zip('YUV', pooled)),
            'pooled_psnr_yuv_db': weighted(pooled),
            'mean_frame_plane_psnr_db': dict(zip('YUV', means)),
            'mean_frame_psnr_yuv_db': weighted(means),
            'exact_reconstruction': all(p['sse'] == 0 for r in rows for p in r['planes'])}


class Reader:
    def __init__(self, data: bytes): self.data, self.pos = data, 0
    def take(self, n: int) -> bytes:
        if n < 0 or n > len(self.data) - self.pos: raise ValueError('truncated bitstream')
        out = self.data[self.pos:self.pos+n]; self.pos += n
        return out
    def unpack(self, fmt): return struct.unpack(fmt, self.take(struct.calcsize(fmt)))
    def integer(self, fmt): return self.unpack(fmt)[0]
    def end(self):
        if self.pos != len(self.data): raise ValueError('trailing bytes / inconsistent nested length')


def access_unit(data: bytes) -> dict:
    r = Reader(data)
    if r.take(4) != b'NVAU': raise ValueError('missing NVAU')
    version = r.integer('<H')
    extra = 0
    if version == 1:
        kind, flags, ids, reserved, w, h, qp, size = r.unpack('<BBHHIIIQ')
        if reserved or flags != (1 if kind == 0 else 0): raise ValueError('invalid NVAU v1 flags')
        identifier = r.take(ids).decode('ascii')
        payload = r.take(size)
        envelope = 32 + ids
    elif version == 2:
        header, flags, kind, pixel, depth, reserved, w, h, qp, doi, poi, deps, sections, cid, pid, mid, reserved2, total = r.unpack('<HIBBBBIIIQQHHHHHHQ')
        if header != 64 or total != len(data) or reserved or reserved2 or depth != 8:
            raise ValueError('invalid NVAU v2 header')
        identifier = r.take(cid).decode('ascii'); r.take(pid + mid + deps * 8)
        entries = [r.unpack('<HHIQ') for _ in range(sections)]
        payloads = [(entry[0], r.take(entry[3])) for entry in entries]
        codec = [body for kind_id, body in payloads if kind_id == 1]
        if len(codec) != 1: raise ValueError('expected one codec-payload section')
        payload = codec[0]
        extra = sum(len(body) for kind_id, body in payloads if kind_id != 1)
        envelope = 64 + cid + pid + mid + deps * 8 + 16 * sections
    else: raise ValueError(f'unsupported NVAU version {version}')
    r.end()
    if identifier != 'dcvcrt': raise ValueError(f'unsupported codec identity {identifier}')
    p = Reader(payload)
    magic = p.take(4)
    if magic not in (b'NVI1', b'NVP1') or len(payload) < 24: raise ValueError('unsupported codec syntax')
    pw, ph, pq, pf = p.unpack('<IIII')
    if (pw, ph, pq) != (w, h, qp) or kind not in (0, 1) or (kind == 0) != (magic == b'NVI1'):
        raise ValueError('inconsistent access-unit/codec fields')
    if not w or not h or w % 2 or h % 2 or pf > (1 if kind == 0 else 3): raise ValueError('invalid codec fields')
    return {'version': version, 'width': w, 'height': h, 'effective_qp': qp,
            'frame_type': 'I' if kind == 0 else 'P', 'two_entropy_coders': bool(pf & 1),
            'use_frame_reference': bool(pf & 2) if kind else True,
            'entropy_bytes': len(payload)-20, 'codec_syntax_bytes': 20,
            'nvau_envelope_bytes': envelope, 'nvau_side_data_bytes': extra,
            'access_unit_bytes': len(data)}


def nvcr_bytes(path: Path) -> dict:
    data = path.read_bytes(); r = Reader(data)
    if r.take(8) != b'NVCS\x01\x00\x00\x00': raise ValueError('unsupported sequence format')
    rows = []
    while r.pos < len(data):
        packet_data = r.take(r.integer('<Q')); p = Reader(packet_data)
        if p.take(4) != b'NVCR': raise ValueError('missing packet magic')
        version, kind, flags, timestamp, count = p.unpack('<HBBQH')
        if version != 1 or flags: raise ValueError('unsupported packet format')
        metadata = 0
        for _ in range(count):
            a, b = p.unpack('<HH'); p.take(a+b); metadata += 4+a+b
        au = access_unit(p.take(p.integer('<Q'))); p.end()
        if kind != (0 if au['frame_type'] == 'I' else 1): raise ValueError('packet/AU frame type mismatch')
        au.update(packet_metadata_bytes=metadata, packet_framing_bytes=26,
                  sequence_record_length_bytes=8, packet_bytes=len(packet_data), timestamp_us=timestamp)
        rows.append(au)
    if not rows: raise ValueError('empty sequence')
    keys = ['entropy_bytes','codec_syntax_bytes','nvau_envelope_bytes','nvau_side_data_bytes',
            'packet_metadata_bytes','packet_framing_bytes','sequence_record_length_bytes']
    components = {k: sum(row[k] for row in rows) for k in keys}
    components['sequence_header_bytes'] = 8
    if sum(components.values()) != len(data): raise ValueError('file-size reconciliation failed')
    entropy = components['entropy_bytes']; overhead = len(data) - entropy
    pixels = sum(row['width'] * row['height'] for row in rows)
    return {'schema':'nvcr.measurement.bytes.v1','format':'NVCS1/Packet1/NVAU',
            'file_sha256':digest(path),'frames':rows,'incremental':components,
            'file_bytes':len(data),'entropy_bytes':entropy,'overhead_bytes':overhead,
            'entropy_bpp':8*entropy/pixels,'file_bpp':8*len(data)/pixels,
            'extra_bpp':8*overhead/pixels,'overhead_bytes_per_access_unit':overhead/len(rows),
            'overhead_fraction_of_file':overhead/len(data),
            'expansion_relative_to_entropy':overhead/entropy if entropy else None}


def summary(values: list[float]) -> dict:
    if not values or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) for v in values):
        raise ValueError('statistics require finite observations; missing/exact values cannot be dropped')
    return {'n':len(values),'mean':statistics.mean(values),
            'sample_std':statistics.stdev(values) if len(values)>1 else None,
            'minimum':min(values),'maximum':max(values),'method':'arithmetic mean; sample SD (ddof=1)',
            'unit_of_analysis':'independent process execution'}


def bd_rate(reference, candidate) -> dict:
    """PCHIP log-rate integration on explicit quality overlap; never sort/repair curves."""
    import scipy
    from scipy.interpolate import PchipInterpolator
    curves = []
    for points in (reference, candidate):
        if len(points) < 4: raise ValueError('at least four RD points required')
        rates, scores = zip(*points)
        if any(not math.isfinite(v) for v in (*rates,*scores)) or min(rates)<=0:
            raise ValueError('RD values must be finite, with positive rates')
        if any(a>=b for a,b in zip(rates,rates[1:])) or any(a>=b for a,b in zip(scores,scores[1:])):
            raise ValueError('RD points must arrive in strictly increasing rate and quality order')
        curves.append((scores, PchipInterpolator(scores, [math.log(r) for r in rates], extrapolate=False)))
    lo=max(c[0][0] for c in curves); hi=min(c[0][-1] for c in curves)
    if not hi>lo: raise ValueError('no positive quality overlap')
    delta=(curves[1][1].integrate(lo,hi)-curves[0][1].integrate(lo,hi))/(hi-lo)
    return {'bd_rate_percent':100*math.expm1(delta),'quality_bounds':[lo,hi],
            'method':'PCHIP log-rate, overlap only','implementation':'scipy '+scipy.__version__}
