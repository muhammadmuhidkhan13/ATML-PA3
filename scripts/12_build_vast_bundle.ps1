$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$FrozenData = Join-Path $ProjectRoot "artifacts\task1\data\metamath-40000-v2"
$ExpectedDataHash = "f8ed6892b608c40579c849c49437ebd6f5097e06377ba3a1605fd9a60f5ce0e6"
$DataFile = Join-Path $FrozenData "examples.jsonl"

if (-not (Test-Path $DataFile -PathType Leaf)) {
    throw "Frozen dataset is missing: $DataFile"
}

$ActualDataHash = (Get-FileHash $DataFile -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualDataHash -ne $ExpectedDataHash) {
    throw "Frozen dataset SHA-256 mismatch. Refusing to build deployment bundle."
}

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$BundlePath = Join-Path $ProjectRoot "ATML-PA3-Vast-A100-$Timestamp.zip"
$StageRoot = Join-Path ([System.IO.Path]::GetTempPath()) "atml-pa3-vast-$([guid]::NewGuid().ToString('N'))"

New-Item -ItemType Directory -Path $StageRoot -Force | Out-Null

try {
    foreach ($Name in @("configs", "scripts", "src", "tests")) {
        $Source = Join-Path $ProjectRoot $Name
        if ($Name -ne "configs" -or (Test-Path -LiteralPath $Source)) {
            Copy-Item -Path $Source -Destination $StageRoot -Recurse
        }
    }

    foreach ($Name in @("AGENTS.md", "TASK1_STATUS.md", "TASK1_TRAINING.md")) {
        $Source = Join-Path $ProjectRoot $Name
        if (Test-Path $Source -PathType Leaf) {
            Copy-Item -Path $Source -Destination $StageRoot
        }
    }

    $StageDataParent = Join-Path $StageRoot "artifacts\task1\data"
    New-Item -ItemType Directory -Path $StageDataParent -Force | Out-Null
    Copy-Item -Path $FrozenData -Destination $StageDataParent -Recurse

    $StageEnvironment = Join-Path $StageRoot "environment"
    New-Item -ItemType Directory -Path $StageEnvironment -Force | Out-Null

    $RequiredPackages = @(
        "transformers",
        "tokenizers",
        "numpy",
        "safetensors",
        "huggingface-hub",
        "pytest"
    )

    $Freeze = & python -m pip freeze
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect the local Python packages."
    }

    $Requirements = foreach ($Package in $RequiredPackages) {
        $PackageRegex = (($Package -split '[-_.]+') | ForEach-Object { [regex]::Escape($_) }) -join '[-_.]+'
        $Pattern = '^' + $PackageRegex + '=='
        $Match = $Freeze |
            Where-Object { $_ -match $Pattern } |
            Select-Object -First 1

        if ($null -eq $Match) {
            throw "Required local package is missing: $Package"
        }

        $Match
    }

    $Requirements | Set-Content `
        -Path (Join-Path $StageEnvironment "requirements-vast.txt") `
        -Encoding utf8

    $LocalVersions = @{
        created_utc = (Get-Date).ToUniversalTime().ToString("o")
        python = (& python -c "import platform; print(platform.python_version())")
        torch = (& python -c "import torch; print(torch.__version__)")
        local_cuda = (& python -c "import torch; print(torch.version.cuda)")
        frozen_data_sha256 = $ActualDataHash
    }
    $LocalVersions | ConvertTo-Json | Set-Content -Path (Join-Path $StageEnvironment "local-versions.json") -Encoding utf8

    Compress-Archive -Path (Join-Path $StageRoot "*") -DestinationPath $BundlePath -CompressionLevel Optimal
}
finally {
    if (Test-Path $StageRoot) {
        Remove-Item -Path $StageRoot -Recurse -Force
    }
}

$BundleHash = (Get-FileHash $BundlePath -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "Vast deployment bundle created successfully."
Write-Host "Bundle: $BundlePath"
Write-Host "Bundle SHA-256: $BundleHash"
Write-Host "Frozen dataset SHA-256: $ActualDataHash"
