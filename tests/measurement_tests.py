#!/usr/bin/env python3
"""Analytic metric and real-process failure fixtures for measurement semantics."""
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import measurement_metrics as metrics
import measurement_campaign as campaign


def wire(entropy=b'abcd',metadata=(),version=1,extra=b''):
    payload=b'NVI1'+struct.pack('<IIII',64,64,21,0)+entropy
    if version==1:
        au=b'NVAU'+struct.pack('<HBBHHIIIQ',1,0,1,6,0,64,64,21,len(payload))+b'dcvcrt'+payload
    else:
        ids=b'dcvcrt'+b'fp16'+b'model'
        sections=1+bool(extra); total=64+len(ids)+16*sections+len(payload)+len(extra)
        au=b'NVAU'+struct.pack('<HHIBBBBIIIQQHHHHHHQ',2,64,1,0,0,8,0,64,64,21,0,0,0,sections,6,4,5,0,total)+ids
        au+=struct.pack('<HHIQ',1,1,1,len(payload))
        if extra: au+=struct.pack('<HHIQ',5,1,0,len(extra))
        au+=payload+extra
    packet=b'NVCR'+struct.pack('<HBBQH',1,0,0,0,len(metadata))
    for a,b in metadata: packet+=struct.pack('<HH',len(a),len(b))+a+b
    packet+=struct.pack('<Q',len(au))+au
    return b'NVCS\x01\x00\x00\x00'+struct.pack('<Q',len(packet))+packet


class MeasurementTests(unittest.TestCase):
    def setUp(self): self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_varying_errors_distinguish_temporal_definitions(self):
        a=self.root/'a'; b=self.root/'b'; a.write_bytes(bytes(12)); b.write_bytes(bytes([1]*6+[3]*6))
        q=metrics.quality(a,b,2,2,2)
        self.assertAlmostEqual(q['pooled_psnr_yuv_db'],10*math.log10(65025/5))
        self.assertAlmostEqual(q['mean_frame_psnr_yuv_db'],10*math.log10(65025/3))
        self.assertEqual([p['sse'] for p in q['per_frame'][1]['planes']],[36,9,9])
        self.assertEqual([p['samples'] for p in q['per_frame'][1]['planes']],[4,1,1])
    def test_plane_weighting(self):
        a=self.root/'a'; b=self.root/'b'; a.write_bytes(bytes(6)); b.write_bytes(bytes([1,1,1,1,2,4]))
        q=metrics.quality(a,b,2,2,1)
        self.assertAlmostEqual(q['pooled_psnr_yuv_db'],(6*10*math.log10(65025)+10*math.log10(65025/4)+10*math.log10(65025/16))/8)
    def test_exact_is_not_capped_or_nonstandard_json(self):
        a=self.root/'a'; a.write_bytes(bytes(6)); q=metrics.quality(a,a,2,2,1)
        self.assertEqual(q['pooled_psnr_yuv_db'],'infinity'); self.assertTrue(q['exact_reconstruction'])
        json.dumps(q,allow_nan=False)
    def test_exact_plane_with_inexact_other_plane_retained(self):
        a=self.root/'a'; b=self.root/'b'; a.write_bytes(bytes(6)); b.write_bytes(bytes([0,0,0,0,1,1]))
        q=metrics.quality(a,b,2,2,1)
        self.assertEqual(q['pooled_psnr_yuv_db'],'infinity'); self.assertFalse(q['exact_reconstruction'])
    def test_invalid_lengths_and_dimensions(self):
        a=self.root/'a'; b=self.root/'b'; a.write_bytes(bytes(12))
        for n in (5,7,11,13):
            b.write_bytes(bytes(n))
            with self.assertRaises(ValueError): metrics.quality(a,b,2,2,2)
        with self.assertRaises(ValueError): metrics.quality(a,a,3,2,1)
    def test_byte_reconciliation_variable_metadata_and_sections(self):
        for version in (1,2):
            for meta in ((),((b'key',b'value'),(b'longer-key',b'x'*41))):
                p=self.root/'stream'; p.write_bytes(wire(metadata=meta,version=version,extra=b'colour' if version==2 else b''))
                b=metrics.nvcr_bytes(p)
                self.assertEqual(sum(b['incremental'].values()),p.stat().st_size)
                self.assertEqual(b['entropy_bytes'],4)
                self.assertEqual(b['incremental']['packet_metadata_bytes'],sum(4+len(k)+len(v) for k,v in meta))
                self.assertEqual(b['file_bytes'],b['entropy_bytes']+b['overhead_bytes'])
    @unittest.skipUnless(os.environ.get('NVCR_WIRE_FIXTURE'),'production writer fixture supplied by CTest')
    def test_actual_cpp_writers(self):
        subprocess.run([os.environ['NVCR_WIRE_FIXTURE'],str(self.root)],check=True)
        paths=list(self.root.glob('*.nvcr')); self.assertEqual(len(paths),4)
        for path in paths:
            b=metrics.nvcr_bytes(path)
            self.assertEqual(b['entropy_bytes'],4)
            self.assertEqual(sum(b['incremental'].values()),path.stat().st_size)
            if path.name.startswith('v2'): self.assertEqual(b['incremental']['nvau_side_data_bytes'],2)

    def test_byte_truncation_and_unsupported_versions(self):
        p=self.root/'s'; data=wire()
        for n in [0,7,15,len(data)-1]:
            p.write_bytes(data[:n])
            with self.assertRaises(ValueError): metrics.nvcr_bytes(p)
        p.write_bytes(data+b'x')
        with self.assertRaises(ValueError): metrics.nvcr_bytes(p)
        p.write_bytes(data.replace(b'NVAU\x01\x00',b'NVAU\x03\x00'))
        with self.assertRaises(ValueError): metrics.nvcr_bytes(p)
    def test_sample_sd_and_invalid_observations(self):
        s=metrics.summary([1.,2.,3.]); self.assertEqual(s['sample_std'],1.); self.assertEqual(s['n'],3)
        for values in ([],[math.nan],[math.inf],[None],[True]):
            with self.assertRaises(ValueError): metrics.summary(values)
        self.assertIsNone(metrics.summary([1])['sample_std'])
    def test_duplicate_or_mixed_observations_rejected(self):
        p=self.root/'observations.jsonl'
        r={'schema':campaign.SCHEMA,'execution_id':'one','fingerprint':'f','job_id':'x','operation':'encode'}
        for rows in ([r,r],[r,{**r,'execution_id':'two','fingerprint':'other'}],[{'schema':'legacy'}]):
            p.write_text('\n'.join(json.dumps(x) for x in rows))
            with self.assertRaises(ValueError): campaign.observations(p)
    def test_child_failure_and_timeout_retained(self):
        for code,timeout,status in [('raise SystemExit(7)',5,'failed'),('import time; time.sleep(10)',.03,'timed_out')]:
            r=campaign.execute([sys.executable,'-c',code],self.root,'child',timeout)
            self.assertEqual(r['status'],status); self.assertGreater(r['process_seconds'],0)
    def test_missing_executable_is_failure(self):
        self.assertEqual(campaign.execute(['/missing/nvcr'],self.root,'x',1)['status'],'failed')
    def test_removed_input_records_failed_cases_without_starting_codec(self):
        m=campaign.load_manifest(campaign.ROOT/'docs/experiments/measurement-campaign.json')
        j=campaign.jobs(m)[0]
        j['sequence']={**j['sequence'],'path':str(self.root/'removed.yuv')}
        rows=campaign.run_job(m,j,self.root,'f',{'evaluated_sha256':'expected','source_sha256':'source'})
        self.assertEqual([r['status'] for r in rows],['failed','failed'])
        self.assertTrue(all('input unavailable' in r['reason'] for r in rows))
        self.assertEqual(len(campaign.observations(self.root/'observations.jsonl')[0]),2)
    def test_empty_run_remains_incomplete(self):
        metrics.write_json(self.root/'run.json',{'expected_operations':2,
            'expected_keys':[['case','encode'],['case','decode']],'fingerprint':'f'})
        result=campaign.analyze(self.root)
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(len(result['missing_observations']),2)
    def test_manifest_keeps_configured_qps_repetitions(self):
        m=campaign.load_manifest(campaign.ROOT/'docs/experiments/measurement-campaign.json')
        self.assertEqual(m['qps'],[0,21,42,63]); self.assertEqual(m['repetitions'],10)
        self.assertEqual(campaign.jobs(m),campaign.jobs(m))
        self.assertEqual(len(campaign.jobs(m)),4*6*3*2*21)
    def test_nvcr_only_keeps_matrix_without_python_jobs(self):
        m=campaign.load_manifest(campaign.ROOT/'docs/experiments/measurement-campaign.json')
        m['implementations']=['nvcr']
        jobs=campaign.jobs(m)
        self.assertEqual({j['implementation'] for j in jobs},{'nvcr'})
        self.assertEqual(2*len(jobs),3024)
        self.assertEqual({j['qp'] for j in jobs},{0,21,42,63})
        self.assertEqual({j['repeat'] for j in jobs if j['mode']=='throughput'},set(range(10)))
    def test_rtx_manifest_keeps_matched_matrix_and_portable_reference_paths(self):
        path=campaign.ROOT/'docs/experiments/measurement-campaign-rtx4070.json'
        raw=json.loads(path.read_text())
        self.assertEqual(raw['implementations'],['nvcr','python'])
        self.assertEqual(raw['qps'],[0,21,42,63]); self.assertEqual(raw['repetitions'],10)
        self.assertEqual(raw['target_profile'],'configs/targets/rtx4070-ubuntu2404.json')
        self.assertFalse(Path(raw['reference_root']).is_absolute())
        self.assertFalse(Path(raw['reference_python']).is_absolute())
        m=campaign.load_manifest(path)
        self.assertEqual(2*len(campaign.jobs(m)),6048)
    def test_profile_line_endings_require_exact_published_digest(self):
        import hashlib
        p=self.root/'model.json'; p.write_bytes(b'{\r\n "model": 1\r\n}\r\n')
        expected=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        bound=campaign.bound_model_profile(p,expected,self.root)
        self.assertEqual(metrics.digest(bound),expected)
        self.assertIn(b'\r\n',p.read_bytes())
        p.write_bytes(b'{\r\n "model": 2\r\n}\r\n')
        with self.assertRaises(ValueError): campaign.bound_model_profile(p,expected,self.root)
    def test_orin_name_alias_retains_hardware_checks(self):
        t=json.loads((campaign.ROOT/'configs/targets/orin-nano-l4t3647.json').read_text())
        d={'device_name':'Orin','architecture':'aarch64','compute_capability_major':8,
           'compute_capability_minor':7,'multiprocessor_count':8,'cuda_runtime_version':12060,
           'tensorrt_version_major':10,'tensorrt_version_minor':3,'tensorrt_version_patch':0}
        self.assertIsNotNone(campaign.measurement_target(t,d)['gpu_name_alias'])
        self.assertEqual(t['gpu']['name'],'Jetson Orin Nano')
        for changed in ({'multiprocessor_count':16},{'device_name':'different GPU'},{'cuda_runtime_version':12070}):
            with self.assertRaises(campaign.legacy.SoftwareXError): campaign.measurement_target(t,{**d,**changed})
    def test_effective_schedule_qp_shift_and_reset(self):
        j={'qp':63,'gop':30,'sequence':{'frames':66,'width':176,'height':144}}
        rows=[]
        for i in range(66):
            intra=i%30==0
            rows.append({'frame_type':'I' if intra else 'P','effective_qp':63+(0 if intra else [0,8,0,4,0,4,0,4][i%8]),
                         'two_entropy_coders':False,'use_frame_reference':bool(intra or i%30==1 or i%64==1)})
        campaign.validate_coding({'frames':rows},j)
        rows[65]['use_frame_reference']=False
        with self.assertRaises(ValueError): campaign.validate_coding({'frames':rows},j)
    def test_interpreter_symlink_keeps_virtual_environment(self):
        m=campaign.load_manifest(campaign.ROOT/'docs/experiments/measurement-campaign.json')
        self.assertIn('/.venv-jetson/bin/python',m['reference_python'])

    def test_analysis_counts_processes_and_reports_missing(self):
        rows=[]; expected=[]
        for i,value in enumerate((10.,20.,30.)):
            job='case-r'+str(i)
            expected.extend([(job,'encode'),(job,'decode')])
            rows.append({'schema':campaign.SCHEMA,'execution_id':str(i),'attempt_id':str(i),
                'job_id':job,'case_id':'case','fingerprint':'f','implementation':'nvcr',
                'operation':'decode','mode':'throughput','status':'passed','process_seconds':value/2,
                'metrics':{'throughput_fps':value,'codec_seconds':100/value}})
        (self.root/'observations.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        metrics.write_json(self.root/'run.json',{'expected_operations':6,'expected_keys':expected,'fingerprint':'f'})
        result=campaign.analyze(self.root)
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(len(result['missing_observations']),3)
        self.assertEqual(result['aggregates'][0]['throughput_fps']['mean'],20.)
        self.assertEqual(result['aggregates'][0]['throughput_fps']['sample_std'],10.)
        self.assertEqual(result['aggregates'][0]['n'],3)

    def test_wrong_timed_frame_count_or_asynchronous_contract_fails(self):
        m={'warmup_frames':2}; seq={'frames':3}
        good={'schema':'nvcr.measurement.operation.v1','timing_contract':metrics.TIMING_CONTRACT,
              'timed_frames':3,'warmup_frames':2,'warmup_policy':'same-session-reset-rewind',
              'synchronization':'device-before-after-each-frame','reference_reset_interval':64,
              'codec_seconds':.5,'throughput_fps':6.}
        campaign.validate_operation(good,seq,m)
        for changed in ({'timed_frames':2},{'synchronization':'enqueue'},{'codec_seconds':math.nan},{'throughput_fps':8.}):
            with self.assertRaises(ValueError): campaign.validate_operation({**good,**changed},seq,m)

    @unittest.skipUnless(importlib.util.find_spec('scipy'),'scipy optional for BD analysis')
    def test_bd_overlap_and_monotonicity(self):
        a=[[1,10],[2,20],[4,30],[8,40]]; b=[[1.1,10],[2.2,20],[4.4,30],[8.8,40]]
        self.assertAlmostEqual(metrics.bd_rate(a,b)['bd_rate_percent'],10.)
        for bad in (list(reversed(b)),[[1,50],[2,60],[3,70],[4,80]],[[1,10],[2,20],[3,math.inf],[4,40]]):
            with self.assertRaises(ValueError): metrics.bd_rate(a,bad)


if __name__=='__main__': unittest.main()
