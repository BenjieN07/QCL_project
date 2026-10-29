import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
try:
    from qcl import ibm_pilot
except ModuleNotFoundError as exc:
    if exc.name not in ('qiskit','qiskit_ibm_runtime'):
        raise
    raise unittest.SkipTest('Optional IBM dependencies are not installed')


class PilotReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root=Path('.tmp').resolve()
        cls.temp_root.mkdir(exist_ok=True)

    def test_hardware_counts_map_to_correct_outputs(self):
        pubs=[SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(get_counts=lambda key=key:{key:1024}))) for key in ('01','10','11')]
        job=Mock(); job.status.return_value='DONE'; job.result.return_value=pubs
        job.metrics.return_value={'usage':{'quantum_seconds':3}}
        service=Mock(); service.job.return_value=job
        service.usage.return_value={'usage_remaining_seconds':591}
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder,patch.object(ibm_pilot,'ROOT',Path(folder)):
            Path(folder,'job.json').write_text(json.dumps({'job_id':'offline-test'}))
            ibm_pilot.retrieve(service)
            report=json.loads(Path(folder,'results.json').read_text())
            self.assertEqual([r['hardware_prediction'] for r in report['rows']],[[-1,1],[1,-1],[-1,-1]])
            self.assertEqual(report['training_updates_on_hardware'],0)
            self.assertEqual(report['usage']['quantum_seconds'],3)

    def test_queued_job_does_not_wait_for_result(self):
        job=Mock(); job.status.return_value='QUEUED'
        service=Mock(); service.job.return_value=job
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder,patch.object(ibm_pilot,'ROOT',Path(folder)):
            Path(folder,'job.json').write_text(json.dumps({'job_id':'offline-test'}))
            ibm_pilot.retrieve(service)
            job.result.assert_not_called()
            self.assertFalse(Path(folder,'results.json').exists())
