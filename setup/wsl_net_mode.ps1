<#
.SYNOPSIS
  Switch WSL2 between the two networking modes this machine needs.

.DESCRIPTION
  On this machine the two WSL2 networking modes fail in opposite directions,
  and both were measured on a fresh Ubuntu-24.04 distro:

    NAT (default, no .wslconfig)
      outbound TCP is dead   -- pypi.org / huggingface.co time out (DNS resolves)
      inbound works          -- Windows localhost:PORT reaches WSL services

    mirrored (.wslconfig networkingMode=mirrored, firewall=false)
      outbound works         -- downloads succeed
      inbound is broken for vLLM specifically: it listens on 0.0.0.0:8000 but
      127.0.0.1 never completes the handshake, from WSL or Windows. A plain
      python http.server on the same port answers fine. Only the WSL interface
      IP reaches vLLM, and only from inside WSL.

  So: use `mirrored` to install packages and download models, then `nat` to
  serve. Each switch runs `wsl --shutdown`, which kills anything running in WSL.

.PARAMETER Mode
  'mirrored' or 'nat'.

.EXAMPLE
  .\tools\setup\wsl_net_mode.ps1 mirrored   # before pip install / hf download
  .\tools\setup\wsl_net_mode.ps1 nat        # before vllm serve
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('mirrored', 'nat')]
    [string]$Mode
)

$ErrorActionPreference = 'Stop'
$cfg = Join-Path $env:USERPROFILE '.wslconfig'

if ($Mode -eq 'mirrored') {
    @'
# Written by setup/wsl_net_mode.ps1 -- mirrored mode.
# Needed for outbound TCP (pip, hf download); NAT drops it on this machine.
# firewall=false because the Hyper-V firewall blocks WSL listeners otherwise.
# Switch back with: tools\setup\wsl_net_mode.ps1 nat
[wsl2]
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
firewall=false
'@ | Set-Content -Path $cfg -Encoding ascii -NoNewline
    Write-Host "wrote $cfg (mirrored)"
} else {
    if (Test-Path $cfg) {
        Remove-Item $cfg
        Write-Host "removed $cfg (back to NAT)"
    } else {
        Write-Host "no $cfg present -- already NAT"
    }
}

Write-Host 'wsl --shutdown (this stops every running WSL process)'
wsl.exe --shutdown

# Give the VM a moment, then report what the distro actually sees.
Start-Sleep -Seconds 3
$ifaces = wsl.exe bash -c "ip -4 -o addr show | awk '{print `$2, `$4}'" 2>$null
Write-Host "interfaces now:"
$ifaces | ForEach-Object { Write-Host "  $_" }

$out = wsl.exe bash -c 'curl -s -o /dev/null -w "%{http_code}" -m 10 https://pypi.org/simple/' 2>$null
Write-Host "outbound (pypi): $out   <- 200 under mirrored, 000 under nat is expected here"
Write-Host "MODE = $Mode"
