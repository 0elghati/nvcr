#!/usr/bin/env python3
"""Controlled measurement preflight, smoke, campaign and independent-run analysis."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from measurement_metrics import (QUALITY_CONTRACT, TIMING_CONTRACT, digest, write_json,
                                 quality, nvcr_bytes, summary, bd_rate)
from measurement_environment import capture, command
import benchmark_softwarex_matrix as legacy

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='nvcr.measurement.observation.v1'


def implementations(m):
    return m.get('implementations',['nvcr','python'])


def bound_model_profile(path,expected,output):
    """Preserve an exact published digest when checkout line endings differ."""
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()==expected: return path
    normalized=raw.replace(b'\r\n',b'\n')
    if hashlib.sha256(normalized).hexdigest()!=expected:
        raise ValueError('published model profile differs beyond CRLF line endings')
    destination=output/'published-model-profile.json'
    destination.write_bytes(normalized)
    return destination


def measurement_target(target,device):
    # The registered build profile uses the module marketing name; CUDA uses Orin.
    effective=copy.deepcopy(target)
    alias=(target['id']=='orin-nano-l4t3647' and target['gpu']['name']=='Jetson Orin Nano'
           and device.get('device_name')=='Orin' and device.get('architecture')=='aarch64'
           and target['gpu']['compute_capability']=='8.7' and target['gpu']['multiprocessor_count']==8)
    if alias: effective['gpu']['name']='Orin'
    result=legacy.validate_test_target(effective,device,'exact')
    return {'validation':result,'gpu_name_alias':'Jetson Orin Nano -> CUDA Orin' if alias else None}


def hash_object(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def load_manifest(path):
    m=json.loads(path.read_text())
    if m.get('schema')!='nvcr.measurement.campaign.v1': raise ValueError('unsupported campaign schema')
    selected=implementations(m)
    if not isinstance(selected,list) or not selected or len(selected)!=len(set(selected)) or any(i not in ['nvcr','python'] for i in selected):
        raise ValueError('invalid implementations')
    for key in ['qps','gops']:
        values=m[key]
        if not values or len(values)!=len(set(values)) or any(type(x)!=int for x in values): raise ValueError(f'invalid {key}')
    if any(not 0<=q<=63 for q in m['qps']) or min(m['gops'])<=0: raise ValueError('invalid coding controls')
    if m['reference_reset_interval']!=64: raise ValueError('native DCVC-RT reference reset is fixed at 64')
    for key in ['repetitions','quality_repetitions','timeout_seconds']:
        if type(m[key])!=int or m[key]<=0: raise ValueError(f'invalid {key}')
    if set(m['modes'])!={'throughput','memory','quality'} or len(m['modes'])!=3: raise ValueError('three distinct modes required')
    for key in ['target_profile','model_profile','nvcr','build_dir','engine_root','reference_root','reference_python']:
        m[key]=os.path.abspath(ROOT/Path(m[key]))
    ids=set()
    for s in m['sequences']:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',s['sequence_id']) or s['sequence_id'] in ids: raise ValueError('invalid/duplicate sequence ID')
        ids.add(s['sequence_id'])
        if any(type(s[k])!=int or s[k]<=0 for k in ['width','height','frames']) or s['width']%2 or s['height']%2: raise ValueError('invalid dimensions/frames')
        if not math.isfinite(s['fps']) or s['fps']<=0 or s['pixel_format']!='yuv420p8': raise ValueError('invalid input format/fps')
        if not 0<=m['warmup_frames']<=s['frames']: raise ValueError('warm-up outside input range')
        s['path']=str((ROOT/Path(s['path'])).resolve())
    if not ids: raise ValueError('empty matrix')
    return m


def engine(m,s):
    try: return legacy.engine_directory(Path(m['engine_root']),s['profile'],{})
    except legacy.SoftwareXError: return Path(m['engine_root'])/s['profile']


def jobs(m):
    rng=random.Random(m['order_seed']); result=[]
    for mode in ['throughput','memory','quality']:
        count=m['quality_repetitions'] if mode=='quality' else m['repetitions']
        for repeat in range(count):
            block=[{'sequence':s,'qp':q,'gop':g,'repeat':repeat,'mode':mode,'implementation':impl}
                   for s in m['sequences'] for q in m['qps'] for g in m['gops'] for impl in implementations(m)]
            rng.shuffle(block); result.extend(block)
    return result


def key(job):
    return f"{job['sequence']['sequence_id']}-q{job['qp']}-g{job['gop']}-{job['mode']}-r{job['repeat']}-{job['implementation']}" + ('-cold' if job.get('warmup_override')==0 else '')


def commands(m,j,directory):
    if 'warmup_override' in j: m={**m,'warmup_frames':j['warmup_override']}
    s=j['sequence']; stream=directory/('stream.nvcr' if j['implementation']=='nvcr' else 'stream.bin')
    reconstruction=directory/'decoded.yuv' if j['mode']=='quality' else Path('/dev/null')
    if j['implementation']=='nvcr':
        base=[m['nvcr']]; shared=['--engine-dir',str(engine(m,s)),'--frames',str(s['frames']),
                              '--warmup-frames',str(m['warmup_frames'])]
        enc=base+['encode','-i',s['path'],'-o',str(stream),'-s',f"{s['width']}x{s['height']}",
                  '-r',str(s['fps']),'--qp',str(j['qp']),'--gop-size',str(j['gop'])]+shared+['--measurement-json',str(directory/'encode.metrics.json')]
        dec=base+['decode','-i',str(stream),'-o',str(reconstruction)]+shared+['--measurement-json',str(directory/'decode.metrics.json')]
    else:
        base=[m['reference_python'],str(ROOT/'scripts/measure_python_reference.py'),'--root',m['reference_root'],
              '--width',str(s['width']),'--height',str(s['height']),'--frames',str(s['frames']),
              '--qp',str(j['qp']),'--gop',str(j['gop']),'--warmup-frames',str(m['warmup_frames'])]
        enc=base+['--operation','encode','--input',s['path'],'--output',str(stream),'--metrics',str(directory/'encode.metrics.json')]
        if j['mode']=='quality': base+=['--validate-reconstruction']
        dec=base+['--operation','decode','--input',str(stream),'--output',str(reconstruction),'--metrics',str(directory/'decode.metrics.json')]
    return enc,dec,stream,reconstruction


def preflight(m,output):
    output.mkdir(parents=True,exist_ok=True)
    env=capture(ROOT,Path(m['reference_python'] if 'python' in implementations(m) else sys.executable),Path(m['reference_root']),Path(m['build_dir']),Path(m['nvcr']))
    write_json(output/'environment.json',env)
    checks=[]; identities={'inputs':{},'engines':{},'checkpoints':{}}
    def check(label,fn):
        try:
            detail=fn(); checks.append({'check':label,'status':'passed','detail':detail}); return detail
        except Exception as e:
            checks.append({'check':label,'status':'failed','reason':f'{type(e).__name__}: {e}'}); return None
    def dependencies():
        import numpy
        return {'numpy':numpy.__version__}
    check('offline metrics dependency',dependencies)
    def build():
        binary=Path(m['nvcr']); cache=Path(m['build_dir'])/'CMakeCache.txt'
        if not binary.is_file() or not os.access(binary,os.X_OK): raise ValueError('NVCR executable unavailable')
        if 'CMAKE_BUILD_TYPE:STRING=Release' not in cache.read_text(): raise ValueError('Release build required')
        help_result=command([binary,'--help'])
        if '--measurement-json' not in help_result.get('stdout',''): raise ValueError('rebuild NVCR: measurement CLI unavailable')
        sources=[ROOT/'cli/main.cpp',ROOT/'cli/CMakeLists.txt']+list((ROOT/'src').rglob('*.cpp'))+list((ROOT/'src').rglob('*.cu'))+list((ROOT/'include').rglob('*.hpp'))
        if any(p.stat().st_mtime>binary.stat().st_mtime for p in sources): raise ValueError('binary predates source: rebuild Release')
        return {'binary_sha256':digest(binary),'build_cache_sha256':digest(cache)}
    check('Release executable',build)
    model=check('model profile',lambda:json.loads(Path(m['model_profile']).read_text()))
    def reference():
        root=Path(m['reference_root'])
        revision=command(['git','-C',root,'rev-parse','HEAD']).get('stdout','').strip()
        dirty=command(['git','-C',root,'status','--porcelain','--untracked-files=no'])
        if revision!=model['upstream']['commit'] or dirty.get('returncode')!=0 or dirty.get('stdout','').strip():
            raise ValueError('reference must be clean pinned upstream source; historical modified wrapper is not accepted')
        for name,spec in model['checkpoints'].items():
            p=root/'checkpoints'/spec['file']; actual=digest(p)
            if actual!=spec['sha256']: raise ValueError(f'{name} checkpoint digest mismatch')
            identities['checkpoints'][name]=actual
        # Python imports an untracked source file ahead of a pinned one only if installed
        # explicitly; record the actual imported extension binaries in the doctor output.
        return {'commit':revision,'wrapper_sha256':digest(ROOT/'scripts/measure_python_reference.py')}
    if 'python' in implementations(m): check('pinned Python source/checkpoints',reference)
    def doctor():
        result=command([m['reference_python'],ROOT/'scripts/measure_python_reference.py','--root',m['reference_root'],
                        '--operation','doctor','--metrics',output/'python-doctor.json'],90)
        if result.get('returncode')!=0: raise ValueError(json.dumps(result))
        return json.loads((output/'python-doctor.json').read_text())
    python_device=check('Python CUDA and required extensions',doctor) if 'python' in implementations(m) else {'status':'not_selected'}
    device=check('NVCR CUDA/TensorRT device',lambda:legacy.load_device_identity(None,0))
    target=check('target profile',lambda:json.loads(Path(m['target_profile']).read_text()))
    if device: check('exact target',lambda:measurement_target(target,device))
    for s in m['sequences']:
        def input_identity(s=s):
            p=Path(s['path']); unit=s['width']*s['height']*3//2; size=p.stat().st_size
            if size%unit or size<s['frames']*unit: raise ValueError('input dimension/frame-count alignment failure')
            data={'source_sha256':digest(p),'evaluated_sha256':digest(p,unit*s['frames']),
                  'source_bytes':size,'evaluated_frames':s['frames'],'start_frame':0,
                  'content_id':s.get('content_id'),'provenance':s.get('provenance',{'status':'unknown'})}
            identities['inputs'][s['sequence_id']]=data; return data
        check('input '+s['sequence_id'],input_identity)
        def artifact(s=s):
            d=engine(m,s); manifest=json.loads((d/'engine_manifest.json').read_text())
            legacy.artifacts.validate_engine_bundle(d)
            identities['engines'][s['profile']]={'path':str(d),'manifest':manifest,
                'sha256':legacy.engine_bundle_digest(d,manifest),
                'export_command':manifest.get('export_command',{'status':'unavailable'}),
                'build_command':manifest.get('build_command',{'status':'unavailable'})}
            if not device: raise ValueError('bundle files inspected; device compatibility unresolved')
            bound=bound_model_profile(Path(m['model_profile']),manifest.get('model_profile_sha256'),output)
            result=legacy.validate_engine_identity(d,manifest,profile=s['profile'],compatibility_class='exact',
                model_profile_path=bound,test_target_path=Path(m['target_profile']),test_target=target,identity=device)
            result['checkout_model_profile_sha256']=digest(Path(m['model_profile']))
            result['model_profile_line_endings_normalized']=bound!=Path(m['model_profile'])
            return result
        check('engine '+s['profile'],artifact)
    if m['execution']['mode']=='container' and not re.fullmatch(r'sha256:[a-f0-9]{64}',m['execution'].get('container_digest') or ''):
        checks.append({'check':'container digest','status':'failed','reason':'immutable container digest required'})
    relevant=list((ROOT/'scripts').glob('measurement*.py'))+[ROOT/'scripts/measure_python_reference.py',ROOT/'cli/main.cpp']
    identity={'manifest':m,'assets':identities,'binary':env['executable'],
              'source_head':env['source']['head'],'source_diff':env['source']['diff'],
              'measurement_sources':{str(p.relative_to(ROOT)):digest(p) for p in relevant},
              'device':device,'python_device':python_device,'build':env['build'],
              'python_dependencies':[c for c in env['commands'] if c['command'][1:]==['-m','pip','freeze']],
              'platform':env['platform'],
              'host_identity':{k:v for k,v in env['host_files'].items() if k not in ['/proc/cpuinfo','/proc/meminfo']},
              'governors':{k:v for k,v in env['sensors_before_or_after_only'].items() if k.endswith('governor')},
              'execution_environment':env['execution_environment'],
              'power_configuration':[c for c in env['commands'] if c['command'][0]=='nvpmodel']}
    report={'schema':'nvcr.measurement.preflight.v1','status':'passed' if all(c['status']=='passed' for c in checks) else 'blocked',
            'checks':checks,'identity':identity,'fingerprint':hash_object(identity),
            'provenance_limitations':['Historical input acquisition/resize commands unavailable; hashes identify current bytes.',
                                      'Historical reference wrapper revision-to-result binding unavailable.']}
    write_json(output/'preflight.json',report)
    return report


def execute(argv,directory,operation,timeout):
    """Reap this exact child with wait4; never use cumulative RUSAGE_CHILDREN."""
    started=time.perf_counter(); proc=None
    with (directory/f'{operation}.stdout').open('wb') as out, (directory/f'{operation}.stderr').open('wb') as err:
        try:
            proc=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True)
            timed_out=False
            while True:
                pid,status,usage=os.wait4(proc.pid,os.WNOHANG)
                if pid: break
                if time.perf_counter()-started>timeout:
                    os.killpg(proc.pid,signal.SIGKILL); pid,status,usage=os.wait4(proc.pid,0); timed_out=True; break
                time.sleep(.01)
            proc.returncode=os.waitstatus_to_exitcode(status)
            err.flush()
            stderr_text=(directory/f'{operation}.stderr').read_text(errors='replace').lower()
            failure_kind=('timeout' if timed_out else 'out_of_memory' if 'out of memory' in stderr_text or 'outofmemory' in stderr_text else 'cuda_misaligned_address' if 'misaligned address' in stderr_text else 'child_exit' if proc.returncode else None)
            return {'status':'timed_out' if timed_out else ('passed' if proc.returncode==0 else 'failed'),
                    'returncode':proc.returncode,'failure_kind':failure_kind,'stdout_path':str(directory/f'{operation}.stdout'),'stderr_path':str(directory/f'{operation}.stderr'),'process_seconds':time.perf_counter()-started,
                    'process_peak_rss_mib':usage.ru_maxrss/1024.,'memory_method':'linux-wait4-ru_maxrss',
                    'command':argv,'reason':failure_kind}
        except OSError as e:
            return {'status':'failed','reason':str(e),'command':argv,'process_seconds':time.perf_counter()-started}
        finally:
            if proc is not None and proc.returncode is None:
                try: os.killpg(proc.pid,signal.SIGKILL); _,status,_=os.wait4(proc.pid,0); proc.returncode=os.waitstatus_to_exitcode(status)
                except ProcessLookupError: pass


def validate_operation(metrics,sequence,m):
    if metrics.get('schema')!='nvcr.measurement.operation.v1' or metrics.get('timing_contract')!=TIMING_CONTRACT:
        raise ValueError('incompatible timing schema/contract')
    if metrics.get('timed_frames')!=sequence['frames'] or metrics.get('warmup_frames')!=m['warmup_frames'] or metrics.get('warmup_policy')!='same-session-reset-rewind':
        raise ValueError('timed frame/warm-up mismatch')
    if metrics.get('reference_reset_interval')!=64 or metrics.get('synchronization')!='device-before-after-each-frame':
        raise ValueError('synchronization/reset mismatch')
    seconds=metrics.get('codec_seconds')
    if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0: raise ValueError('invalid completed codec interval')
    if not math.isclose(metrics['throughput_fps'],sequence['frames']/seconds,rel_tol=1e-9): raise ValueError('invalid FPS denominator')


def validate_coding(accounting,j):
    s=j['sequence']; rows=accounting['frames']
    if len(rows)!=s['frames']: raise ValueError('stream frame count mismatch')
    shifts=[0,8,0,4,0,4,0,4]
    for i,r in enumerate(rows):
        intra=i%j['gop']==0; q=j['qp']+(0 if intra else shifts[i%8])
        if r['frame_type']!=('I' if intra else 'P') or r['effective_qp']!=q: raise ValueError('effective frame schedule/QP mismatch')
        if r['two_entropy_coders']!=(s['width']*s['height']>=1280*720): raise ValueError('entropy coder mode mismatch')
        if not intra and r['use_frame_reference']!=bool(i%j['gop']==1 or i%64==1): raise ValueError('feature-reference reset mismatch')
        if 'width' in r and (r['width'],r['height'])!=(s['width'],s['height']): raise ValueError('stream dimension mismatch')


def run_job(m,j,root,fingerprint,input_identity):
    if 'warmup_override' in j: m={**m,'warmup_frames':j['warmup_override']}
    execution_id=str(uuid.uuid4()); directory=root/'raw'/key(j)/execution_id; directory.mkdir(parents=True)
    enc,dec,stream,reconstruction=commands(m,j,directory); rows=[]; s=j['sequence']; accounting=None
    evaluated_bytes=s['frames']*s['width']*s['height']*3//2
    input_error=None
    try:
        if digest(Path(s['path']),evaluated_bytes)!=input_identity['evaluated_sha256']:
            input_error='input changed since preflight'
    except (OSError,ValueError) as error:
        input_error=f'input unavailable since preflight: {error}'
    for operation,argv in [('encode',enc),('decode',dec)]:
        row={'schema':SCHEMA,'execution_id':execution_id+'-'+operation,'attempt_id':execution_id,
             'job_id':key(j),'case_id':f"{s['sequence_id']}-q{j['qp']}-g{j['gop']}",
             'fingerprint':fingerprint,'implementation':j['implementation'],'operation':operation,'mode':j['mode'],
             'repeat':j['repeat'],'sequence':s['sequence_id'],'content_id':s.get('content_id'),
             'qp':j['qp'],'gop':j['gop'],'frames':s['frames'],'width':s['width'],'height':s['height'],'fps_metadata':s['fps'],
             'input_path':s['path'],'input_sha256':input_identity['evaluated_sha256'],'source_sha256':input_identity['source_sha256'],'raw_directory':str(directory),'timing_contract':TIMING_CONTRACT,
             'instrumentation':('reconstruction validation; offline quality' if j['mode']=='quality' else 'no polling, quality or verbose tracing inside interval'),
             'units':{'codec_seconds':'s','process_seconds':'s','throughput_fps':'frames/s','process_peak_rss_mib':'MiB'}}
        if input_error: row.update(status='failed',reason=input_error)
        elif operation=='decode' and rows[0]['status']!='passed': row.update(status='skipped',reason='encode did not succeed')
        else:
            row.update(execute(argv,directory,operation,m['timeout_seconds']))
            if j['mode']!='memory': row.pop('process_peak_rss_mib',None)
            if row['status']=='passed':
                try:
                    if digest(Path(s['path']),evaluated_bytes)!=input_identity['evaluated_sha256']: raise ValueError('input changed during execution')
                    metric=json.loads((directory/f'{operation}.metrics.json').read_text()); validate_operation(metric,s,m)
                    row['metrics']=metric
                    if operation=='encode':
                        accounting=nvcr_bytes(stream) if j['implementation']=='nvcr' else metric['bytes']
                        validate_coding(accounting,j)
                        if accounting['file_sha256']!=digest(stream) or accounting['file_bytes']!=stream.stat().st_size:
                            raise ValueError('stream identity mismatch')
                        row['bytes']=accounting; write_json(directory/'bytes.json',accounting)
                    if operation=='decode' and j['mode']=='quality':
                        q=quality(Path(s['path']),reconstruction,s['width'],s['height'],s['frames'])
                        write_json(directory/'quality.json',q); row['quality']=q
                except Exception as e: row.update(status='failed',reason=f'metric validation: {e}')
        rows.append(row)
        with (root/'observations.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
    # Large successful products are dispensable after hashing/evaluation. Failures remain auditable.
    if all(r['status']=='passed' for r in rows):
        stream.unlink(missing_ok=True)
        if reconstruction!=Path('/dev/null'): reconstruction.unlink(missing_ok=True)
    return rows


def observations(path):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if any(r.get('schema')!=SCHEMA for r in rows): raise ValueError('legacy/aggregate rows cannot enter this analysis')
    if len({r['execution_id'] for r in rows})!=len(rows): raise ValueError('duplicate execution identity')
    if len({r['fingerprint'] for r in rows})>1: raise ValueError('mixed build/input/environment fingerprints')
    latest={}
    for row in rows: latest[(row['job_id'],row['operation'])]=row
    return rows,list(latest.values())


def analyze(root):
    all_rows,rows=observations(root/'observations.jsonl'); groups={}; failures=[r for r in rows if r['status']!='passed']
    for r in rows:
        if r['status']!='passed': continue
        group=(r['case_id'],r['implementation'],r['operation'],r['mode'])
        groups.setdefault(group,[]).append(r)
    aggregates=[]
    for group,values in sorted(groups.items()):
        case,impl,op,mode=group
        item={'case_id':case,'implementation':impl,'operation':op,'mode':mode,'n':len(values)}
        if mode=='throughput':
            for field in ['codec_seconds','throughput_fps']:
                item[field]=summary([v['metrics'][field] for v in values])
            item['process_seconds']=summary([v['process_seconds'] for v in values])
        elif mode=='memory': item['process_peak_rss_mib']=summary([v['process_peak_rss_mib'] for v in values])
        elif op=='decode': item['quality']=[v['quality'] for v in values]
        if op=='encode': item['byte_observations']=[{'file_bpp':v['bytes']['file_bpp'],'entropy_bpp':v['bytes']['entropy_bpp']} for v in values]
        aggregates.append(item)
    run=json.loads((root/'run.json').read_text()); expected=run['expected_operations']
    expected_keys={tuple(x) for x in run['expected_keys']}
    actual_keys={(r['job_id'],r['operation']) for r in rows}
    if actual_keys-expected_keys: raise ValueError('observations contain unplanned cases')
    if any(r['fingerprint']!=run['fingerprint'] for r in rows): raise ValueError('run/observation identity mismatch')
    mismatched_attempts=[]
    for job_id in {r['job_id'] for r in rows}:
        attempts={r['attempt_id'] for r in rows if r['job_id']==job_id}
        if len(attempts)!=1: mismatched_attempts.append(job_id)
    reset_checks=[]
    if run.get('smoke'):
        for impl in implementations(run.get('manifest',{})):
            for op in ['encode','decode']:
                candidates=[r for r in rows if r['implementation']==impl and r['mode']=='quality' and r['operation']==op and r['status']=='passed']
                digests=[r['bytes']['file_sha256'] if op=='encode' else r['quality']['decoded_sha256'] for r in candidates]
                reset_checks.append({'implementation':impl,'operation':op,'status':'passed' if len(digests)==2 and len(set(digests))==1 else 'unverified_or_failed','comparison':'cold versus warmed/reset output SHA256'})
    complete=actual_keys==expected_keys and len(rows)==expected and not failures and not mismatched_attempts and all(c['status']=='passed' for c in reset_checks)
    report={'schema':'nvcr.measurement.analysis.v1','status':'complete' if complete else 'incomplete',
            'expected_operations':expected,'observed_operations':len(rows),'failed_or_skipped':len(failures),
            'missing_observations':[{'job_id':j,'operation':op,'status':'incomplete','reason':'no completed observation recorded'} for j,op in sorted(expected_keys-actual_keys)],
            'mismatched_attempts':mismatched_attempts,'reset_checks':reset_checks,
            'failed_attempts_retained':sum(r['status']!='passed' for r in all_rows),
            'statistical_method':'mean and sample SD, independent process executions within condition; no inferred pairing',
            'aggregates':aggregates,'failures':failures}
    write_json(root/'analysis.json',report)
    # RD inputs use the same decoded integer metric and both byte definitions.
    rd=[]
    by_job={r['job_id']:r for r in rows if r['status']=='passed' and r['mode']=='quality' and r['operation']=='encode'}
    for r in rows:
        if (r['status']=='passed' and r['mode']=='quality' and r['operation']=='decode'
                and r['job_id'] in by_job and r['attempt_id']==by_job[r['job_id']]['attempt_id']):
            b=by_job[r['job_id']]['bytes']; rd.append({'sequence':r['sequence'],'content_id':r['content_id'],'implementation':r['implementation'],
                'gop':r['gop'],'qp':r['qp'],'input_sha256':r['input_sha256'],'frames':r['frames'],'width':r['width'],'height':r['height'],'fingerprint':r['fingerprint'],'quality_contract':QUALITY_CONTRACT,'psnr_yuv_db':r['quality']['pooled_psnr_yuv_db'],
                'entropy_bpp':b['entropy_bpp'],'file_bpp':b['file_bpp']})
    write_json(root/'rd-points.json',{'schema':'nvcr.measurement.rd.v1','points':rd,'analysis':'raw points; no implicit sorting or extrapolation'})
    return report


def validate_smoke_evidence(path,pre):
    run=json.loads((path/'run.json').read_text())
    result=analyze(path)  # Recompute from observations, never trust a status-only summary.
    if not run.get('smoke') or result['status']!='complete': raise ValueError('smoke is not complete')
    old=run['preflight_identity']; current=pre['identity']
    for field in ['binary','source_head','source_diff','measurement_sources','device','python_device','build','execution_environment','power_configuration','python_dependencies','platform','host_identity','governors']:
        if old[field]!=current[field]: raise ValueError('smoke incompatible with current '+field)
    if old['assets']['checkpoints']!=current['assets']['checkpoints']: raise ValueError('smoke checkpoint mismatch')
    for profile,identity in old['assets']['engines'].items():
        if current['assets']['engines'].get(profile)!=identity: raise ValueError('smoke engine mismatch')
    for sequence,identity in old['assets']['inputs'].items():
        if current['assets']['inputs'].get(sequence,{}).get('source_sha256')!=identity['source_sha256']:
            raise ValueError('smoke source content mismatch')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['capture','preflight','run','smoke','analyze','bd-rate'])
    ap.add_argument('--manifest',type=Path,default=ROOT/'docs/experiments/measurement-campaign.json')
    ap.add_argument('--implementation',action='append',choices=['nvcr','python'],help='select implementation(s); defaults to manifest or both')
    ap.add_argument('--engine-root',type=Path,help='use an existing installed engine root')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--smoke-evidence',type=Path,help='complete compatible smoke package required for full launch')
    ap.add_argument('--curves',type=Path,help='BD input JSON: reference and candidate [[rate,quality],...] in increasing order')
    args=ap.parse_args(argv); root=args.output.resolve()
    if args.action=='analyze':
        report=analyze(root); print(json.dumps({k:v for k,v in report.items() if k not in ['aggregates','failures']})); return 0 if report['status']=='complete' else 1
    if args.action=='bd-rate':
        curves=json.loads(args.curves.read_text())
        if curves.get('quality_contract')!=QUALITY_CONTRACT or curves.get('rate_definition') not in ['entropy_bpp','file_bpp']:
            raise ValueError('BD analysis requires common quality contract and an explicit common rate definition')
        result=bd_rate(curves['reference'],curves['candidate']); result.update(quality_contract=QUALITY_CONTRACT,rate_definition=curves['rate_definition'])
        write_json(root,result); return 0
    m=load_manifest(args.manifest)
    if args.implementation:
        if len(args.implementation)!=len(set(args.implementation)): raise ValueError('duplicate implementation selection')
        m['implementations']=args.implementation
    if args.engine_root: m['engine_root']=str(args.engine_root.resolve())
    if args.action=='smoke':
        m=copy.deepcopy(m); m['sequences']=[m['sequences'][0]]; m['sequences'][0]['frames']=3
        m['qps']=[m['qps'][0]]; m['gops']=[next(g for g in m['gops'] if g>1)]
        m['repetitions']=1; m['warmup_frames']=2; m['timeout_seconds']=180
    if args.action=='capture':
        write_json(root,capture(ROOT,Path(m['reference_python'] if 'python' in implementations(m) else sys.executable),Path(m['reference_root']),Path(m['build_dir']),Path(m['nvcr']))); return 0
    if root.exists() and any(root.iterdir()) and not args.resume: raise ValueError('output must be new/empty; compatible resume requires --resume')
    root.mkdir(parents=True,exist_ok=True)
    preflight_dir=root/'preflight'/str(uuid.uuid4())
    pre=preflight(m,preflight_dir); schedule=jobs(m)
    if args.action=='smoke':
        schedule += [{**j,'warmup_override':0} for j in list(schedule) if j['mode']=='quality']
    if (root/'run.json').exists():
        previous=json.loads((root/'run.json').read_text())
        if previous['fingerprint']!=pre['fingerprint']: raise ValueError('incompatible resume: original plan preserved; build/input/environment/contract changed')
    elif (root/'observations.jsonl').exists():
        raise ValueError('orphaned observations without run identity; refusing reuse')
    plan=[{'job_id':key(j),'encode':commands(m,j,root/'raw'/key(j)/'ATTEMPT')[0],
           'decode':commands(m,j,root/'raw'/key(j)/'ATTEMPT')[1]} for j in schedule]
    write_json(root/'plan.json',{'manifest':m,'jobs':plan,'expected_operations':2*len(plan)})
    (root/'commands.txt').write_text('\n'.join(shlex.join(p[op]) for p in plan for op in ['encode','decode'])+'\n')
    print(f"Preflight: {pre['status']}; {len(plan)*2} operations; resolved commands: {root/'commands.txt'}")
    for c in pre['checks']:
        if c['status']!='passed': print(c['check']+': '+c['reason'])
    if args.action=='preflight' or args.dry_run or pre['status']!='passed': return 0 if pre['status']=='passed' else 1
    if args.action=='run':
        if args.smoke_evidence is None: raise ValueError('full launch requires --smoke-evidence from a successful bounded smoke')
        validate_smoke_evidence(args.smoke_evidence,pre)
    run_path=root/'run.json'
    if run_path.exists():
        previous=json.loads(run_path.read_text())
        if previous['fingerprint']!=pre['fingerprint']: raise ValueError('incompatible resume: build/input/environment/contract changed')
    else: write_json(run_path,{'fingerprint':pre['fingerprint'],'preflight_identity':pre['identity'],'manifest':m,'expected_operations':2*len(schedule),'expected_keys':[(key(j),op) for j in schedule for op in ['encode','decode']],'smoke':args.action=='smoke'})
    completed=set()
    if (root/'observations.jsonl').exists():
        _,existing=observations(root/'observations.jsonl')
        for job in schedule:
            matches=[r for r in existing if r['job_id']==key(job)]
            if len(matches)==2 and len({r['attempt_id'] for r in matches})==1 and all(r['status']=='passed' for r in matches): completed.add(key(job))
    for index,j in enumerate(schedule):
        if key(j) in completed: continue
        print(f'[{index+1}/{len(schedule)}] {key(j)}',flush=True)
        run_job(m,j,root,pre['fingerprint'],pre['identity']['assets']['inputs'][j['sequence']['sequence_id']])
    write_json(root/'environment-after.json',capture(ROOT,Path(m['reference_python'] if 'python' in implementations(m) else sys.executable),Path(m['reference_root']),Path(m['build_dir']),Path(m['nvcr'])))
    report=analyze(root)
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,OSError,KeyError,legacy.SoftwareXError) as error:
        print(f'measurement campaign: {error}',file=sys.stderr); raise SystemExit(2)
