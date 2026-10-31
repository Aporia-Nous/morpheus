$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "=== Morpheus CUDA hardware validation ==="

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js is required."
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found. Install/repair the NVIDIA driver first."
}

$nvcc = $env:MORPHEUS_NVCC
if (-not $nvcc -and $env:CUDA_PATH) {
    $candidate = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
    if (Test-Path $candidate) { $nvcc = $candidate }
}
if (-not $nvcc) {
    $cmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($cmd) { $nvcc = $cmd.Source }
}
if (-not $nvcc) {
    throw "nvcc was not found. Install the NVIDIA CUDA Toolkit or set MORPHEUS_NVCC."
}

Write-Host "NVCC: $nvcc"
& $nvcc --version
if ($LASTEXITCODE -ne 0) { throw "nvcc --version failed." }

$gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1).Trim()
Write-Host "GPU: $gpuName"

$arch = $env:MORPHEUS_CUDA_ARCH
if (-not $arch) {
    try {
        $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim()
        if ($cap -match '^([0-9]+)\.([0-9]+)$') {
            $arch = "sm_$($Matches[1])$($Matches[2])"
        }
    } catch {}
}
if (-not $arch) { $arch = "sm_89" }
Write-Host "CUDA arch: $arch"

Write-Host "Running model-free Morpheus GPU API validation..."
node gpu_test.js
if ($LASTEXITCODE -ne 0) { throw "gpu_test.js failed." }

$outDir = Join-Path $env:TEMP "morpheus-gpu-validation"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outExe = Join-Path $outDir "gpu_compute.exe"

$env:MORPHEUS_NVCC = $nvcc
Write-Host "Building real CUDA executable..."
node gpu_build.js examples/gpu_compute.morph --arch $arch -o $outExe
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $outExe)) {
    throw "CUDA build failed or did not produce $outExe"
}

Write-Host "Running real CUDA executable..."
$output = (& $outExe 2>&1 | Out-String)
Write-Host $output
if ($LASTEXITCODE -ne 0) { throw "GPU executable exited with code $LASTEXITCODE" }

if ($output -notmatch 'A x B:\s*58,64,139,154') {
    throw "Matrix multiplication validation failed. Expected A x B: 58,64,139,154"
}
if ($output -notmatch 'GPU compute completed\.') {
    throw "GPU completion marker was not observed."
}

Write-Host "PASS: Morpheus executed CUDA tensor operations on $gpuName ($arch)."
Write-Host "Executable: $outExe"
