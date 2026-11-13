param(
    [Parameter(Mandatory=$true)][string]$Package,
    [Parameter(Mandatory=$true)][string]$HostName,
    [string]$UserName="root", [int]$Port=22,
    [string]$RemoteDir="/workspace/atml-task2-upload",
    [switch]$ExecuteTransfer
)
$ErrorActionPreference="Stop"
if ($HostName -notmatch '^[A-Za-z0-9.-]+$' -or $UserName -notmatch '^[A-Za-z0-9_-]+$' -or $Port -lt 1 -or $Port -gt 65535) {
    throw 'Invalid SSH host, user, or port.'
}
if ($RemoteDir -notmatch '^/[A-Za-z0-9_/-]*task2[A-Za-z0-9_/-]*$' -or $RemoteDir.Contains('..')) {
    throw 'Use a distinct Task2 upload directory with simple characters.'
}
$packagePath=(Resolve-Path -LiteralPath $Package).Path
$manifest=Get-Content -LiteralPath (Join-Path $packagePath 'transfer_manifest.json') -Raw | ConvertFrom-Json
if (-not $ExecuteTransfer) {
    Write-Host "Validation only: $($manifest.parts.Count) chunks would go to $HostName under $RemoteDir. No connection made."
    exit 0
}
$target="$UserName@$HostName"
& ssh -p $Port $target "mkdir -p '$RemoteDir'"
if ($LASTEXITCODE -ne 0) { throw 'Could not create the separate Task2 upload directory.' }
foreach ($part in $manifest.parts) {
    if ($part.name -notmatch '^part_[0-9]+\.bin$') { throw 'Invalid chunk filename.' }
    $local=Join-Path $packagePath $part.name
    if ((Get-FileHash -LiteralPath $local -Algorithm SHA256).Hash.ToLower() -ne $part.sha256) { throw 'Local chunk hash differs.' }
    $existing=& ssh -p $Port $target "if test -f '$RemoteDir/$($part.name)'; then sha256sum '$RemoteDir/$($part.name)'; fi"
    if ($existing -and $existing.StartsWith($part.sha256)) { continue }
    & scp -P $Port $local "${target}:$RemoteDir/$($part.name)"
    if ($LASTEXITCODE -ne 0) { throw 'Chunk transfer interrupted. Rerun to skip verified complete chunks.' }
    $verified=& ssh -p $Port $target "sha256sum '$RemoteDir/$($part.name)'"
    if (-not $verified.StartsWith($part.sha256)) { throw 'Transferred chunk hash differs.' }
}
foreach ($name in @('transfer_manifest.json','task2_unpack.py')) {
    & scp -P $Port (Join-Path $packagePath $name) "${target}:$RemoteDir/$name"
    if ($LASTEXITCODE -ne 0) { throw 'Bootstrap transfer failed.' }
}
Write-Host 'Task2 chunks transferred. No extraction, installation, or training was run.'
