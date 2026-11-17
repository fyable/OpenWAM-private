import hashlib
import io
import json
import tarfile
import ast
import subprocess
import sys
from pathlib import Path

from benchmarks.libero.paraflow_video_worker import instrument, count_video_stages

from benchmarks.libero.reproduce_paraflow_110 import prepare, video_launcher


def test_prepare_frozen_source_and_record_selection(tmp_path):
    packet = tmp_path / 'repo/results/20261017-hardware-ablation/evidence'
    evidence = packet / 'openwam-formal-host2-v6r'
    evidence.mkdir(parents=True)
    content = b'print("historical")\n'
    identity = {'source': {'scripts/diagnostics/benchmark_wam_cumulative.py': hashlib.sha256(content).hexdigest()},
                'assets': {}, 'options': {}, 'overlay': [], 'additional_overlay': []}
    (evidence / 'identity.json').write_text(json.dumps(identity))
    with tarfile.open(packet / 'frozen-seven-row-source.tar.gz', 'w:gz') as archive:
        for name in ['wam/scripts/diagnostics/benchmark_wam_cumulative.py', 'wam/../escape.py', 'first3/skip.py']:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    source, manifest = prepare(tmp_path / 'output', tmp_path / 'repo')
    assert (source / 'scripts/diagnostics/benchmark_wam_cumulative.py').read_bytes() == content
    assert not (tmp_path / 'output/escape.py').exists()
    assert manifest['status'] == 'prepared_not_run'
    assert 'C1-C5 excluded' in manifest['selection']


def test_video_instrumentation_retains_setup_and_replaces_campaign():
    source = '''
def main():
    setting = 10
    cases = {"baseline": "must not run"}
    raise RuntimeError("old campaign")
main()
'''
    calls = []
    namespace = {'run_video_experiment': lambda context: calls.append(context['setting'])}
    exec(compile(instrument(source), '<test>', 'exec'), namespace)
    assert calls == [10]
    assert not any(isinstance(node, ast.Raise) for node in ast.walk(instrument(source)))


def test_video_launcher_ignores_shadow_package_and_passes_global_save(tmp_path):
    shadow = tmp_path / 'benchmarks'
    shadow.mkdir()
    (shadow / '__init__.py').write_text('raise RuntimeError("shadow package imported")\n')
    script = tmp_path / 'campaign.py'
    script.write_text('''
def save(*args):
    return "global save"
def main():
    args = None
    cases = {}
main()
''')
    worker = Path(__file__).resolve().parents[2] / 'benchmarks/libero/paraflow_video_worker.py'
    launcher = video_launcher(worker).replace(
        'worker["launch"]',
        'worker["launch"].__globals__.__setitem__("run_video_experiment", '
        'lambda context: print(context["save"]())) or worker["launch"]',
    )
    result = subprocess.run([sys.executable, '-c', launcher, str(script)],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'global save'


def test_video_stage_counter_tracks_native_path_and_restores():
    from types import SimpleNamespace

    class Stages:
        def full(self, videos):
            return videos

    model = object()
    stage = Stages()
    stage.arch = model
    original = Stages.full
    counts = {'full_video_calls': 0, 'future_latents': 0}
    video = SimpleNamespace(ndim=5, shape=(10, 32, 3, 24, 20))
    with count_video_stages(Stages, model, counts):
        assert stage.full(video) is video
        stage.full(SimpleNamespace(ndim=5, shape=(10, 32, 1, 24, 20)))
        stage.arch = object()
        stage.full(video)
    assert counts == {'full_video_calls': 1, 'future_latents': 2}
    assert Stages.full is original
