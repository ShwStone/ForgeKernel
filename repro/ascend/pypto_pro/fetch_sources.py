#!/usr/bin/env python3
"""Fetch inspected upstream examples with pinned hashes; never execute them.

Downloaded code retains its upstream license. Sources remain outside the
tracked tree. A documentation change requires explicit manual re-inspection
and updating the pin rather than silently running different code.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.request

FA_URL = ('https://pypto.gitcode.com/_downloads/'
          '31e4d98a385f5cf80b8d3457c63e9760/'
          'test_fa_perf_tkv_preload_dn_vf_tail_tilegroup.py')
PIPELINE_URL = ('https://pypto.gitcode.com/guide/programming_guide/pro/'
                'advanced_programming/auto_parallel_pipeline.html')
FA_SHA256 = 'f1002fd5fa48bda353895f9986dee363d659cf5e0334f1cddf68b06eae7cbdf7'
PIPELINE_SHA256 = '3d0436bfa5fbc40521e3ff6d524f6ab2f83264a6b86f6d38b8e4ae7d723a0f10'


class CodeBlocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.blocks = []
        self.buffer = []

    def handle_starttag(self, tag, attrs):
        if tag == 'pre':
            self.inside = True
            self.buffer = []

    def handle_endtag(self, tag):
        if tag == 'pre' and self.inside:
            self.blocks.append(''.join(self.buffer))
            self.inside = False

    def handle_data(self, data):
        if self.inside:
            self.buffer.append(data)


def extract_pipeline_code(html):
    parser = CodeBlocks()
    parser.feed(html)
    matches = [b for b in parser.blocks
               if 'N_ITER = 4' in b and 'def pipeline_demo_kernel' in b]
    if len(matches) != 1:
        raise ValueError(f'expected one complete pipeline example, found {len(matches)}')
    return matches[0].encode('utf-8')


def verify(data, expected):
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f'source changed: sha256={actual}, expected={expected}; inspect before updating pin')
    return data


def fetch(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError('upstream source exceeds 4 MiB limit')
    return data


def serial_variant(data):
    old = b'@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=2))'
    if data.count(old) != 1:
        raise ValueError('expected one preload=2 kernel decorator')
    return data.replace(old, old.replace(b'preload=2', b'preload=0'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    fa = verify(fetch(FA_URL), FA_SHA256)
    pipeline = verify(extract_pipeline_code(fetch(PIPELINE_URL).decode('utf-8')),
                      PIPELINE_SHA256)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = {'official_flash_attention.py': fa,
             'official_auto_pipeline.py': pipeline,
             'auto_pipeline_serial.py': serial_variant(pipeline)}
    for name, data in files.items():
        path = out / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f'refusing to overwrite changed file: {path}')
    for name, data in files.items():
        (out / name).write_bytes(data)
    manifest = dict(sources=[dict(file=n, sha256=hashlib.sha256(b).hexdigest())
                             for n, b in files.items()],
                    flash_attention_url=FA_URL, pipeline_url=PIPELINE_URL,
                    serial_variant='only preload=2 -> preload=0 in kernel decorator')
    (out / 'sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
