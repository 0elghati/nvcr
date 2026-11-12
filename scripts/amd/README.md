# AMD / MIGraphX environment preparation

Runbook for issue [#224](https://github.com/0elghati/nvcr/issues/224) — prepare the
AMD environment for the MIGraphX path of the C10 milestone. The scripts here
are read-only probes; the system install itself is a sequence of commands the
operator runs by hand on the AMD laptop.

Target in this milestone: **Radeon RX 6700 / 6700 XT (gfx1031)** on **Ubuntu
24.04 LTS** (dual-boot, not WSL). gfx1031 is outside AMD's official ROCm
support matrix, so the `HSA_OVERRIDE_GFX_VERSION=10.3.0` workaround is part of
the recorded state, not a bug.

## Scripts

| Script | Role |
|---|---|
| `check_rocm_support.sh` | Read-only probe: OS, kernel, GPU, ROCm version, user groups, ONNX Runtime providers. Safe to source or run. |
| `verify_migraphx.py` | Builds a trivial ONNX graph and runs it through `MIGraphXExecutionProvider`. Fails if the EP is missing, if ORT silently falls back to CPU, or if numerics are wrong. |

Neither script installs anything.

## Procedure on the AMD laptop

1. **Confirm the OS and record a baseline.** From a fresh Ubuntu 24.04
   terminal, in a `nvcr` checkout of the `feat/amd-migraphx-task-224` branch:

   ```bash
   lsb_release -a
   uname -r
   ./scripts/amd/check_rocm_support.sh | tee /tmp/amd-env-before.txt
   ```

   Expect `ROCm version: not installed` and `onnxruntime: not importable` at
   this point. Keep the output.

2. **Install ROCm 6.x.** Use AMD's current amdgpu-install package for Ubuntu
   24.04 (Noble). AMD updates the exact URL and version frequently; take them
   from the current page:
   <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html>

   The install sequence is roughly:

   ```bash
   sudo apt update
   wget https://repo.radeon.com/amdgpu-install/<rocm-version>/ubuntu/noble/amdgpu-install_<pkg-version>_all.deb
   sudo apt install ./amdgpu-install_<pkg-version>_all.deb
   sudo apt update
   sudo amdgpu-install --usecase=graphics,rocm
   sudo usermod -a -G render,video "$USER"
   ```

   Reboot after this step so the `amdgpu` kernel module and the new group
   membership take effect.

3. **Record the HSA override.** For gfx1031, add to `~/.bashrc` (or a
   per-shell export):

   ```bash
   export HSA_OVERRIDE_GFX_VERSION=10.3.0
   ```

   Re-source and confirm `env | grep HSA_OVERRIDE`.

4. **Install ONNX Runtime with the MIGraphX EP.** Options, in order of
   preference:

   - A ROCm/MIGraphX-enabled wheel from AMD or the ONNX Runtime project (the
     `onnxruntime-rocm` channel, when available for the ROCm version
     installed).
   - A local build of ONNX Runtime with `--use_migraphx`, following
     <https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html>.

   In either case, do it in a Python virtualenv so system Python stays clean:

   ```bash
   python3 -m venv ~/.venv/nvcr-amd
   source ~/.venv/nvcr-amd/bin/activate
   pip install --upgrade pip wheel
   pip install numpy onnx
   pip install <onnxruntime-wheel-for-this-ROCm>
   ```

5. **Re-run the probe and verify.** In the same virtualenv:

   ```bash
   ./scripts/amd/check_rocm_support.sh | tee /tmp/amd-env-after.txt
   python3 scripts/amd/verify_migraphx.py --json | tee /tmp/amd-verify.json
   ```

   `check_rocm_support.sh` should now report a real `ROCm version`, the
   detected GPU, `amdgpu loaded: yes`, both groups `yes`, `HSA_OVERRIDE` set,
   and `MIGraphXExecutionProvider` in the providers list.

   `verify_migraphx.py` must exit 0 and report `"ok": true`, with
   `session_providers` resolving to `["MIGraphXExecutionProvider"]` alone
   (not falling back to CPU).

6. **Capture the results.** Paste the three captured files
   (`/tmp/amd-env-before.txt`, `/tmp/amd-env-after.txt`, `/tmp/amd-verify.json`)
   as a comment on issue [#224](https://github.com/0elghati/nvcr/issues/224),
   or commit them under `evidence/task-224/` on this branch and push. Include:

   - The exact ROCm package version installed and the install source URL
     (AMD's repo, a specific wheel, or a local build commit).
   - The exact `onnxruntime` wheel or build (name, version, source).
   - Any step that deviated from this runbook and why.
   - Any risks or warnings encountered (kernel module warnings, DKMS build
     failures, HSA override needed for a different reason, etc.).

7. **Close #224** once the two captured outputs show a working MIGraphX EP
   on gfx1031 with `HSA_OVERRIDE_GFX_VERSION=10.3.0` recorded.

## If MIGraphX does not come up on gfx1031

Document the failure and stop. Do not work around it silently. gfx1031 being
unsupported is a known risk recorded in the C10 milestone text; a hard failure
here changes which GPU the milestone can land on, not which workaround to try
next. Bring the captured output back to the issue for the next decision.
