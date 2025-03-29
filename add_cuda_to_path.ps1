# Add CUDA to PATH script
# Run as Administrator to add CUDA to the system PATH environment variable

# Check if the script is running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "This script needs to be run as Administrator. Right-click the script and select 'Run as Administrator'."
    exit
}

# Define function to find CUDA installations
function Find-CudaInstallations {
    $cudaVersions = @()
    $programFiles = $env:ProgramFiles
    $cudaBasePath = Join-Path $programFiles "NVIDIA GPU Computing Toolkit\CUDA"
    
    if (Test-Path $cudaBasePath) {
        Get-ChildItem $cudaBasePath -Directory | ForEach-Object {
            if ($_.Name -match "v(\d+\.\d+)") {
                $version = $matches[1]
                $cudaVersions += @{
                    Version = $version
                    Path = $_.FullName
                }
            }
        }
    }
    
    return $cudaVersions
}

# Find CUDA installations
$cudaInstallations = Find-CudaInstallations

if ($cudaInstallations.Count -eq 0) {
    Write-Host "No CUDA installations found. Please install CUDA from https://developer.nvidia.com/cuda-downloads"
    exit
}

# Display found installations
Write-Host "Found CUDA installations:"
for ($i = 0; $i -lt $cudaInstallations.Count; $i++) {
    Write-Host "$($i+1). CUDA $($cudaInstallations[$i].Version) at $($cudaInstallations[$i].Path)"
}

# Ask which version to use if multiple installations found
$selectedIndex = 0
if ($cudaInstallations.Count -gt 1) {
    $validInput = $false
    while (-not $validInput) {
        $selection = Read-Host "Enter the number of the CUDA version to add to PATH"
        if ($selection -match "^\d+$" -and [int]$selection -ge 1 -and [int]$selection -le $cudaInstallations.Count) {
            $selectedIndex = [int]$selection - 1
            $validInput = $true
        } else {
            Write-Host "Invalid selection. Please enter a number between 1 and $($cudaInstallations.Count)"
        }
    }
}

# Get the selected CUDA installation
$selectedCuda = $cudaInstallations[$selectedIndex]
Write-Host "Selected CUDA $($selectedCuda.Version) at $($selectedCuda.Path)"

# Paths to add
$binPath = Join-Path $selectedCuda.Path "bin"
$libnvvpPath = Join-Path $selectedCuda.Path "libnvvp"
$cuptiPath = Join-Path $selectedCuda.Path "extras\CUPTI\lib64"

$pathsToAdd = @()

if (Test-Path $binPath) {
    $pathsToAdd += $binPath
}

if (Test-Path $libnvvpPath) {
    $pathsToAdd += $libnvvpPath
}

if (Test-Path $cuptiPath) {
    $pathsToAdd += $cuptiPath
}

if ($pathsToAdd.Count -eq 0) {
    Write-Host "No valid paths found in the selected CUDA installation."
    exit
}

# Get the current PATH
$currentPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine)
$pathArray = $currentPath -split ";"

# Check if paths are already in PATH
$pathsToAddFiltered = @()
foreach ($path in $pathsToAdd) {
    if (-not ($pathArray -contains $path)) {
        $pathsToAddFiltered += $path
    }
}

if ($pathsToAddFiltered.Count -eq 0) {
    Write-Host "All CUDA paths are already in the system PATH."
    exit
}

# Add new paths to PATH
$newPath = $currentPath
foreach ($path in $pathsToAddFiltered) {
    $newPath += ";$path"
    Write-Host "Adding to PATH: $path"
}

# Set the new PATH
[System.Environment]::SetEnvironmentVariable("Path", $newPath, [System.EnvironmentVariableTarget]::Machine)

# Also set CUDA_PATH environment variable
[System.Environment]::SetEnvironmentVariable("CUDA_PATH", $selectedCuda.Path, [System.EnvironmentVariableTarget]::Machine)
Write-Host "Set CUDA_PATH to $($selectedCuda.Path)"

Write-Host "CUDA paths have been added to the system PATH."
Write-Host "Please restart your computer for the changes to take effect." 