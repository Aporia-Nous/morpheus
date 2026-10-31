$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "=== Morpheus CUDA hardware validation ==="

if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "Node.js is required." }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw "nvidia-smi was not found. Install/repair the NVIDIA driver first." }

$nvcc = $env:MORPHEUS_NVCC
if (-not $nvcc -and $env:CUDA_PATH) {
    $candidate = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
    if (Test-Path $candidate) { $nvcc = $candidate }
}
if (-not $nvcc) {
    $cmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($cmd) { $nvcc = $cmd.Source }
}
if (-not $nvcc) { throw "nvcc was not found. Install the NVIDIA CUDA Toolkit or set MORPHEUS_NVCC." }

Write-Host "NVCC: $nvcc"
& $nvcc --version
if ($LASTEXITCODE -ne 0) { throw "nvcc --version failed." }

$gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1).Trim()
Write-Host "GPU: $gpuName"

$arch = $env:MORPHEUS_CUDA_ARCH
if (-not $arch) {
    try {
        $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim()
        if ($cap -match '^([0-9]+)\.([0-9]+)$') { $arch = "sm_$($Matches[1])$($Matches[2])" }
    } catch {}
}
if (-not $arch) { $arch = "sm_89" }
Write-Host "CUDA arch: $arch"

Write-Host "Running model-free Morpheus GPU API validation..."
node gpu_test.js
if ($LASTEXITCODE -ne 0) { throw "gpu_test.js failed." }

$outDir = Join-Path $env:TEMP "morpheus-gpu-validation"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$computeExe = Join-Path $outDir "gpu_compute.exe"
$aiExe = Join-Path $outDir "gpu_ai_primitives.exe"

$env:MORPHEUS_NVCC = $nvcc

Write-Host "Building real CUDA compute executable..."
node gpu_build.js examples/gpu_compute.morph --arch $arch --runtime v2 -o $computeExe
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $computeExe)) { throw "CUDA compute build failed or did not produce $computeExe" }

Write-Host "Running real CUDA compute executable..."
$computeOutput = (& $computeExe 2>&1 | Out-String)
Write-Host $computeOutput
if ($LASTEXITCODE -ne 0) { throw "GPU compute executable exited with code $LASTEXITCODE" }
if ($computeOutput -notmatch 'A x B:\s*58,64,139,154') { throw "Matrix multiplication validation failed. Expected A x B: 58,64,139,154" }
if ($computeOutput -notmatch 'GPU compute completed\.') { throw "GPU compute completion marker was not observed." }

Write-Host "Building transformer-oriented CUDA executable..."
node gpu_build.js examples/gpu_ai_primitives.morph --arch $arch --runtime v2 -o $aiExe
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $aiExe)) { throw "CUDA AI primitive build failed or did not produce $aiExe" }

Write-Host "Running transformer-oriented CUDA executable..."
$aiOutput = (& $aiExe 2>&1 | Out-String)
Write-Host $aiOutput
if ($LASTEXITCODE -ne 0) { throw "GPU AI executable exited with code $LASTEXITCODE" }

foreach ($marker in @('softmax:', 'rmsnorm:', 'weighted rmsnorm:', 'silu:', 'gelu:', 'rope:', 'gathered embeddings:', 'causal attention:', 'AI GPU primitives completed.')) {
    if ($aiOutput -notmatch [regex]::Escape($marker)) { throw "AI GPU validation marker missing: $marker" }
}
if ($aiOutput -notmatch 'live tensors before cleanup:') { throw "GPU memory-accounting output was not observed." }
if ($aiOutput -notmatch 'released tensors:') { throw "GPU bulk-release output was not observed." }

Write-Host "PASS: Morpheus executed general and transformer-oriented CUDA operations on $gpuName ($arch)."
Write-Host "Compute executable: $computeExe"
Write-Host "AI executable: $aiExe"
