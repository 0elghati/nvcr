"""Read-only environment and source capture; never changes clocks or power state."""
from __future__ import annotations
import os
import platform
import subprocess
from pathlib import Path
from measurement_metrics import digest


def command(argv, timeout=20):
    try:
        p=subprocess.run([str(x) for x in argv],capture_output=True,text=True,timeout=timeout)
        return {'command':[str(x) for x in argv],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,
                'status':'available' if p.returncode==0 else 'unavailable'}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {'command':[str(x) for x in argv],'status':'unavailable','reason':str(e)}


def read(path):
    try: return Path(path).read_text(errors='replace').replace('\x00','').strip()
    except (OSError, TypeError) as e: return {'status':'unavailable','reason':str(e)}


def source(root):
    return {'root':str(root),'head':command(['git','-C',root,'rev-parse','HEAD']),
            'status':command(['git','-C',root,'status','--porcelain']),
            'diff':command(['git','-C',root,'diff','HEAD','--']),
            'submodules':command(['git','-C',root,'submodule','status'])}


def capture(root: Path, python: Path, reference: Path, build: Path, binary: Path):
    files={str(p):read(p) for p in [Path('/etc/os-release'),Path('/etc/nv_tegra_release'),
           Path('/proc/device-tree/model'),Path('/proc/cpuinfo'),Path('/proc/meminfo'),Path('/proc/driver/nvidia/version')]}
    sensors={}
    for pattern in ['/sys/class/thermal/thermal_zone*/temp','/sys/class/thermal/thermal_zone*/type',
                    '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor',
                    '/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq',
                    '/sys/class/devfreq/*/cur_freq','/sys/class/devfreq/*/governor']:
        import glob
        for name in glob.glob(pattern): sensors[name]=read(name)
    commands=[['uname','-a'],['lscpu'],['c++','--version'],['cmake','--version'],
              ['nvidia-smi','-q'],['nvpmodel','-q','--verbose'],['jetson_clocks','--show'],
              ['dpkg-query','-W','nvidia-jetpack','nvidia-l4t-core','libnvinfer10','cuda-cudart-12-6'],
              [python,'-m','pip','freeze'],['ldd',binary]]
    build_files={}
    for rel in ['CMakeCache.txt','cli/CMakeFiles/nvcr_cli.dir/flags.make','cli/CMakeFiles/nvcr_cli.dir/link.txt',
                'CMakeFiles/nvcr.dir/flags.make']:
        p=build/rel; build_files[rel]={'content':read(p),'sha256':digest(p) if p.is_file() else None}
    env_keys=['CUDA_VISIBLE_DEVICES','CUDA_MODULE_LOADING','LD_LIBRARY_PATH','PYTHONPATH','OMP_NUM_THREADS',
              'SUPPRESS_CUSTOM_KERNEL_WARNING','NVCR_TENSORRT_LOW_MEMORY_MODE','NVCR_VERIFY_ENCODER_RECONSTRUCTION']
    return {'schema':'nvcr.measurement.environment.v1','platform':platform.platform(),
            'source':source(root),'reference_source':source(reference),'host_files':files,
            'sensors_before_or_after_only':sensors or {'status':'unavailable'},
            'commands':[command(c) for c in commands], 'build':build_files,
            'executable':{'path':str(binary),'sha256':digest(binary) if binary.is_file() else None},
            'execution_environment':{k:os.environ.get(k) for k in env_keys},
            'memory_contract':'Linux wait4 ru_maxrss, KiB / 1024 = MiB, whole isolated process including initialization and warm-up',
            'gpu_memory':'unavailable as a common quantity; not inferred from RSS or summed on Jetson',
            'thermal_throttle_history':'unavailable; snapshots cannot establish absence of throttling',
            'historical_reference':historical_reference()}


def historical_reference():
    root=Path('/home/oelghati/DCVC-RT')
    names=['runner/load.py','runner/executor.py','test_video_energy.py','test_video.py',
           'src/layers/cuda_inference.py','src/layers/layers.py']
    return {'status':'historical revision-to-record binding unavailable; current files only',
            'root':str(root),'head':command(['git','-C',root,'rev-parse','HEAD']),
            'current_file_sha256':{name:digest(root/name) if (root/name).is_file() else None for name in names},
            'changes_from_pinned':'See assessment; modified wrapper is not executed by the new campaign'}
