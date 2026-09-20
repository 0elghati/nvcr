#!/usr/bin/env python3
# Model orchestration adapted from Microsoft DCVC test_video.py.
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""New measurement wrapper for pinned Microsoft DCVC-RT; not the historical wrapper.

Model calls, frame scheduling and output serialization follow pinned test_video.py.
Only measurement boundaries/warm-up and explicit entropy-coder setting differ.
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import time
from pathlib import Path
from measurement_metrics import TIMING_CONTRACT, digest, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--operation', choices=['encode','decode','doctor'], required=True)
    ap.add_argument('--input', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--metrics', type=Path, required=True)
    ap.add_argument('--width', type=int, default=176)
    ap.add_argument('--height', type=int, default=144)
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--qp', type=int, default=0)
    ap.add_argument('--gop', type=int, default=100)
    ap.add_argument('--warmup-frames', type=int, default=10)
    ap.add_argument('--validate-reconstruction',action='store_true')
    ap.add_argument('--device', type=int, default=0)
    a=ap.parse_args()
    sys.path.insert(0,str(a.root.resolve()))
    import torch
    import numpy as np
    from src.layers import cuda_inference
    from src.layers.cuda_inference import replicate_pad
    from src.models.image_model import DMCI
    from src.models.video_model import DMC
    from src.utils.common import get_state_dict, set_torch_env
    from src.utils.stream_helper import (SPSHelper, NalType, write_sps, write_ip,
                                        read_header, read_sps_remaining, read_ip_remaining)
    from src.utils.transforms import ycbcr420_to_444_np, yuv_444_to_420
    import MLCodec_extensions_cpp
    import inference_extensions_cuda
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; CPU fallback forbidden')
    if not cuda_inference.CUSTOMIZED_CUDA_INFERENCE: raise RuntimeError('required custom CUDA inference extension unavailable')
    torch.cuda.set_device(a.device)
    torch.cuda.synchronize(a.device)
    info={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(a.device),
          'custom_cuda':True,'entropy_extension':str(MLCodec_extensions_cpp.__file__),
          'cuda_extension':str(inference_extensions_cuda.__file__),
          'entropy_extension_sha256':digest(Path(MLCodec_extensions_cpp.__file__)),
          'cuda_extension_sha256':digest(Path(inference_extensions_cuda.__file__)),
          'precision':'float16','eval':True,'gradients':False}
    if a.operation=='doctor':
        write_json(a.metrics,info); return
    if not a.input or not a.output or min(a.width,a.height,a.frames,a.gop)<=0 or a.width%2 or a.height%2 or not 0<=a.qp<=63 or not 0<=a.warmup_frames<=a.frames:
        raise ValueError('invalid operation arguments')
    set_torch_env()
    device=torch.device('cuda',a.device)
    load_start=time.perf_counter()
    i_net=DMCI(); p_net=DMC()
    for net,name in [(i_net,'image'),(p_net,'video')]:
        net.load_state_dict(get_state_dict(str(a.root/'checkpoints'/f'cvpr2025_{name}.pth.tar')))
        net.to(device); net.eval(); net.update(None); net.half()
    torch.cuda.synchronize(device)
    model_load_seconds=time.perf_counter()-load_start
    two_coders=a.width*a.height>=1280*720
    i_net.set_use_two_entropy_coders(two_coders); p_net.set_use_two_entropy_coders(two_coders)
    padding_r,padding_b=DMCI.get_padding_size(a.height,a.width,16)
    area=a.width*a.height; frame_bytes=area*3//2
    index_map=[0,1,0,2,0,2,0,2]
    bitstream=a.input.read_bytes() if a.operation=='decode' else None
    total_seconds=0.; accounting=[]; sps_bytes_total=0
    with torch.no_grad():
        for warming,count in [(True,a.warmup_frames),(False,a.frames)]:
            if count==0: continue
            p_net.clear_dpb(); p_net.set_curr_poc(0)
            helper=SPSHelper(); last_qp=0
            source=io.BytesIO(bitstream) if bitstream is not None else a.input.open('rb')
            sink=Path('/dev/null').open('wb') if warming else a.output.open('wb')
            try:
                for index in range(count):
                    if a.operation=='encode':
                        raw=source.read(frame_bytes)
                        if len(raw)!=frame_bytes: raise ValueError('source truncated')
                    torch.cuda.synchronize(device)
                    started=time.perf_counter()
                    if a.operation=='encode':
                        y=np.frombuffer(raw[:area],dtype=np.uint8).reshape(1,a.height,a.width)
                        uv=np.frombuffer(raw[area:],dtype=np.uint8).reshape(2,a.height//2,a.width//2)
                        x=torch.from_numpy(ycbcr420_to_444_np(y,uv)).to(device).to(torch.float32)/255.0
                        x=replicate_pad(x.unsqueeze(0).half(),padding_b,padding_r)
                        intra=index%a.gop==0
                        ada=0
                        if intra:
                            qp=a.qp; encoded=i_net.compress(x,qp)
                            p_net.clear_dpb(); p_net.add_ref_frame(None,encoded['x_hat'])
                        else:
                            ada=int(index%64==1)
                            if ada: p_net.prepare_feature_adaptor_i(last_qp)
                            qp=p_net.shift_qp(a.qp,index_map[index%8])
                            encoded=p_net.compress(x,qp); last_qp=qp
                        sps={'sps_id':-1,'height':a.height,'width':a.width,'ec_part':int(two_coders),'use_ada_i':ada}
                        sid,new=helper.get_sps_id(sps); sps['sps_id']=sid
                        packet=io.BytesIO()
                        sps_bytes=write_sps(packet,sps) if new else 0
                        write_ip(packet,intra,sid,qp,encoded['bit_stream'])
                        raw_out=packet.getvalue()
                    else:
                        header=read_header(source)
                        while header['nal_type']==NalType.NAL_SPS:
                            sps=read_sps_remaining(source,header['sps_id']); helper.add_sps_by_id(sps)
                            header=read_header(source)
                        sps=helper.get_sps_by_id(header['sps_id'])
                        if sps is None or (sps['width'],sps['height'])!=(a.width,a.height):
                            raise ValueError('missing SPS or dimension mismatch')
                        qp,payload=read_ip_remaining(source)
                        if header['nal_type']==NalType.NAL_I:
                            decoded=i_net.decompress(payload,sps,qp)
                            p_net.clear_dpb(); p_net.add_ref_frame(None,decoded['x_hat'])
                        elif header['nal_type']==NalType.NAL_P:
                            if sps['use_ada_i']: p_net.reset_ref_feature()
                            decoded=p_net.decompress(payload,sps,qp)
                        else: raise ValueError('unexpected frame NAL')
                        x=decoded['x_hat'][:,:,:a.height,:a.width]
                        # Pinned evaluator's serialized output convention: Y rounds, UV truncates.
                        if a.validate_reconstruction and not torch.isfinite(x).all().item():
                            raise ValueError('non-finite floating reconstruction in quality pass')
                        y,uv=yuv_444_to_420(x)
                        y=torch.clamp(y*255,0,255).round().to(torch.uint8).cpu().numpy()
                        uv=torch.clamp(uv*255,0,255).to(torch.uint8).cpu().numpy()
                        raw_out=y.tobytes()+uv.tobytes()
                    torch.cuda.synchronize(device)
                    elapsed=time.perf_counter()-started
                    sink.write(raw_out)
                    if not warming:
                        total_seconds+=elapsed
                        if a.operation=='encode':
                            accounting.append({'frame_index':index,'frame_type':'I' if intra else 'P','effective_qp':qp,
                                'two_entropy_coders':two_coders,'use_frame_reference':bool(intra or index%a.gop==1 or ada),
                                'entropy_bytes':len(encoded['bit_stream']),
                                'codec_syntax_bytes':len(raw_out)-sps_bytes-len(encoded['bit_stream']),
                                'sequence_header_bytes':sps_bytes})
                            sps_bytes_total+=sps_bytes
                if a.operation=='decode' and not warming and source.read(1): raise ValueError('unexpected trailing bitstream')
            finally:
                source.close(); sink.close()
            torch.cuda.synchronize(device)
            p_net.clear_dpb(); p_net.set_curr_poc(0)
    result={'schema':'nvcr.measurement.operation.v1','timing_contract':TIMING_CONTRACT,
            'synchronization':'device-before-after-each-frame','warmup_policy':'same-session-reset-rewind',
            'warmup_frames':a.warmup_frames,'timed_frames':a.frames,'codec_seconds':total_seconds,
            'throughput_fps':a.frames/total_seconds,'reference_reset_interval':64,
            'initialization_seconds':model_load_seconds,'reconstruction_finite_check':a.validate_reconstruction,'backend':info,
            'upstream_reported_quality':{'status':'not_collected','reason':'clean operation; common offline decoded-output quality is primary'},
            'serialization':'pinned test_video.py Y round-to-even after half scaling; UV truncation; 2x2 chroma average; crop before subsampling'}
    if a.operation=='encode':
        file_bytes=a.output.stat().st_size
        entropy=sum(r['entropy_bytes'] for r in accounting)
        syntax=sum(r['codec_syntax_bytes'] for r in accounting)
        if entropy+syntax+sps_bytes_total!=file_bytes: raise ValueError('Python file size reconciliation failed')
        result['bytes']={'schema':'nvcr.measurement.bytes.v1','format':'upstream-SPS-IP',
                         'file_sha256':digest(a.output),'file_bytes':file_bytes,'entropy_bytes':entropy,
                         'frames':accounting,'incremental':{'entropy_bytes':entropy,'codec_syntax_bytes':syntax,'sequence_header_bytes':sps_bytes_total},
                         'file_bpp':file_bytes*8/(a.frames*area),'entropy_bpp':entropy*8/(a.frames*area),
                         'extra_bpp':(file_bytes-entropy)*8/(a.frames*area),
                         'overhead_bytes':file_bytes-entropy,'overhead_bytes_per_access_unit':(file_bytes-entropy)/a.frames,
                         'overhead_fraction_of_file':(file_bytes-entropy)/file_bytes,
                         'expansion_relative_to_entropy':(file_bytes-entropy)/entropy}
    write_json(a.metrics,result)


if __name__=='__main__': main()
