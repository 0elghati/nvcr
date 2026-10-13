"""Selected-pass scope and packaged-binary identity regression checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import measurement_campaign as campaign
from measurement_metrics import digest

class LaptopMeasurements(unittest.TestCase):
    def test_only_timing_pass_without_changing_commands_or_counts(self):
        m=campaign.load_manifest(campaign.ROOT/'docs/experiments/measurement-campaign-rtx4070.json')
        m['implementations']=['nvcr']; full=campaign.jobs(m)
        m['modes']=['throughput']; selected=campaign.jobs(m)
        expected={campaign.key(j) for j in full if j['mode']=='throughput'}
        self.assertEqual({campaign.key(j) for j in selected},expected)
        self.assertEqual(len(selected)*2,1440)
        for j in selected:
            encode,decode,*_=campaign.commands(m,j,Path('/results/run'))
            for cmd in [encode,decode]:
                self.assertEqual(cmd[cmd.index('--warmup-frames')+1],'10')
                self.assertEqual(cmd[cmd.index('--frames')+1],'100')
                self.assertIn('--measurement-json',cmd)
            self.assertNotIn('--quality-metrics',decode)

    def test_packaged_binary_hash_and_launch_digest_must_match(self):
        with tempfile.TemporaryDirectory() as d:
            binary=Path(d)/'nvcr'; binary.write_bytes(b'existing release binary')
            image={'digest':'sha256:'+'a'*64,'id':'sha256:'+'b'*64,
                   'source_revision':'c'*40,'version':'1.0.7','binary_sha256':digest(binary)}
            m={'execution':{'mode':'container','container_digest':image['digest'],'image':image}}
            self.assertEqual(campaign.container_release(m,binary)['binary_sha256'],digest(binary))
            m['execution']['container_digest']='sha256:'+'d'*64
            with self.assertRaisesRegex(ValueError,'digest mismatch'): campaign.container_release(m,binary)
            m['execution']['container_digest']=image['digest']; binary.write_bytes(b'different binary')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): campaign.container_release(m,binary)

    def test_invalid_or_empty_passes_fail(self):
        path=campaign.ROOT/'docs/experiments/measurement-campaign-rtx4070.json'
        raw=json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json'
            for modes in [[],['throughput','throughput'],['unknown']]:
                raw['modes']=modes; p.write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError,'invalid measurement modes'): campaign.load_manifest(p)

if __name__=='__main__': unittest.main()
