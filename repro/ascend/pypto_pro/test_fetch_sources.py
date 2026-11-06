import hashlib
import unittest
from fetch_sources import extract_pipeline_code, serial_variant, verify


class SourceTests(unittest.TestCase):
    def test_matching_hash(self):
        data = b'hello'
        self.assertEqual(verify(data, hashlib.sha256(data).hexdigest()), data)

    def test_changed_source_rejected(self):
        with self.assertRaises(ValueError):
            verify(b'changed', hashlib.sha256(b'original').hexdigest())

    def test_extracts_complete_example(self):
        html = '<pre>ignore</pre><pre>N_ITER = 4\ndef pipeline_demo_kernel():\n    pass\n</pre>'
        self.assertEqual(extract_pipeline_code(html),
                         b'N_ITER = 4\ndef pipeline_demo_kernel():\n    pass\n')

    def test_ambiguous_or_missing_example_rejected(self):
        for html in ('<pre>nothing</pre>', '<pre>N_ITER = 4\ndef pipeline_demo_kernel(): pass</pre>'*2):
            with self.assertRaises(ValueError):
                extract_pipeline_code(html)

    def test_serial_changes_only_decorator(self):
        before = b'@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=2))\ndef kernel(): pass\n'
        self.assertEqual(serial_variant(before), before.replace(b'preload=2', b'preload=0'))
        with self.assertRaises(ValueError):
            serial_variant(b'other source')


if __name__ == '__main__':
    unittest.main()
