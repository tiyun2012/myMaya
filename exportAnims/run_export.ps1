# Define the path to mayapy.exe (update if necessary)
$MAYA_PY = "C:\Program Files\Autodesk\Maya2024\bin\mayapy.exe"

# Load required .NET assembly for Windows Forms
Add-Type -AssemblyName System.Windows.Forms

function Select-FolderDialog {
    $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
    if ($folderBrowser.ShowDialog() -eq 'OK') {
        return $folderBrowser.SelectedPath
    }
    return $null
}

function Ensure-DirectoryExists {
    param([string]$Path)
    if (-not (Test-Path -Path $Path)) {
        try { New-Item -ItemType Directory -Path $Path -Force | Out-Null }
        catch {
            [System.Windows.Forms.MessageBox]::Show("Failed to create directory: $Path`n$($_.Exception.Message)", 'Error')
            throw
        }
    }
}

function Invoke-MayapyScript {
    param(
        [string]$ScriptPath,
        [string]$InputFile,
        [string]$OutputPath
    )
    try { & $MAYA_PY $ScriptPath $InputFile $OutputPath }
    catch { Write-Error "Error executing mayapy on file ${InputFile}: $($_.Exception.Message)" }
}

$form = New-Object System.Windows.Forms.Form
$form.Text = "Path Selector"
$form.Size = New-Object System.Drawing.Size(400, 300)
$form.StartPosition = 'CenterScreen'

$labelSource = New-Object System.Windows.Forms.Label
$labelSource.Text = 'Source Path:'
$labelSource.Location = New-Object System.Drawing.Point(10, 20)
$labelSource.AutoSize = $true
$labelSource.ForeColor = 'Blue'
$labelSource.Cursor = [System.Windows.Forms.Cursors]::Hand
$labelSource.Add_Click({
    $path = Select-FolderDialog
    if ($path) { $textSource.Text = $path }
})
$form.Controls.Add($labelSource)

$textSource = New-Object System.Windows.Forms.TextBox
$textSource.Location = New-Object System.Drawing.Point(100, 20)
$textSource.Size = New-Object System.Drawing.Size(200, 20)
$form.Controls.Add($textSource)

$buttonBrowseSource = New-Object System.Windows.Forms.Button
$buttonBrowseSource.Text = '...'
$buttonBrowseSource.Location = New-Object System.Drawing.Point(310, 20)
$buttonBrowseSource.Add_Click({
    $path = Select-FolderDialog
    if ($path) { $textSource.Text = $path }
})
$form.Controls.Add($buttonBrowseSource)

$labelOutput = New-Object System.Windows.Forms.Label
$labelOutput.Text = 'Output Path:'
$labelOutput.Location = New-Object System.Drawing.Point(10, 60)
$labelOutput.AutoSize = $true
$labelOutput.ForeColor = 'Blue'
$labelOutput.Cursor = [System.Windows.Forms.Cursors]::Hand
$labelOutput.Add_Click({
    $path = Select-FolderDialog
    if ($path) { $textOutput.Text = $path }
})
$form.Controls.Add($labelOutput)

$textOutput = New-Object System.Windows.Forms.TextBox
$textOutput.Location = New-Object System.Drawing.Point(100, 60)
$textOutput.Size = New-Object System.Drawing.Size(200, 20)
$form.Controls.Add($textOutput)

$buttonBrowseOutput = New-Object System.Windows.Forms.Button
$buttonBrowseOutput.Text = '...'
$buttonBrowseOutput.Location = New-Object System.Drawing.Point(310, 60)
$buttonBrowseOutput.Add_Click({
    $path = Select-FolderDialog
    if ($path) { $textOutput.Text = $path }
})
$form.Controls.Add($buttonBrowseOutput)

$buttonExport = New-Object System.Windows.Forms.Button
$buttonExport.Text = 'Export'
$buttonExport.Location = New-Object System.Drawing.Point(10, 100)
$buttonExport.Add_Click({
    $sourcePath = $textSource.Text
    $outputPath = $textOutput.Text

    if (-not (Test-Path -Path $sourcePath)) {
        [System.Windows.Forms.MessageBox]::Show("Source path does not exist!", 'Error')
        return
    }

    Ensure-DirectoryExists -Path $outputPath

    $SCRIPT_PATH = Join-Path -Path $PSScriptRoot -ChildPath "exportAnims.py"

    Get-ChildItem -Path $sourcePath -Filter "*.ma" | ForEach-Object -Process {
        $file = $_.FullName
        Write-Host "------> Start Exporting ${file}" -ForegroundColor Yellow
        Invoke-MayapyScript -ScriptPath $SCRIPT_PATH -InputFile $file -OutputPath $outputPath
        Write-Host "Exporting completed for ${file} <-------" -ForegroundColor Green
    }

    [System.Windows.Forms.MessageBox]::Show("Exporting completed!", 'Export')
})
$form.Controls.Add($buttonExport)

$buttonGetFramesInfo = New-Object System.Windows.Forms.Button
$buttonGetFramesInfo.Text = 'Get Frames Range'
$buttonGetFramesInfo.Location = New-Object System.Drawing.Point(120, 100)
$buttonGetFramesInfo.Add_Click({
    $sourcePath = $textSource.Text
    $outputPath = $textOutput.Text

    if (-not (Test-Path -Path $sourcePath)) {
        [System.Windows.Forms.MessageBox]::Show("Source path does not exist!", 'Error')
        return
    }

    Ensure-DirectoryExists -Path $outputPath

    $SCRIPT_PATH = Join-Path -Path $PSScriptRoot -ChildPath "getAnimInfo.py"
    $data = @()
    $index = 1

    Get-ChildItem -Path $sourcePath -Filter "*.ma" | ForEach-Object -Process {
        $file = $_.FullName
        Write-Host "------> Start Getting Frames Info for ${file}" -ForegroundColor Yellow
        $info = Invoke-MayapyScript -ScriptPath $SCRIPT_PATH -InputFile $file -OutputPath $outputPath

        $infoStr = $info | Out-String

        if ($infoStr -match "startFrame=', ([\d.]+).*?endFrame=', ([\d.]+).*?frames=', ([\d.]+)") {
            $startFrame = [double]$matches[1]
            $endFrame = [double]$matches[2]
            $numberOfFrames = [double]$matches[3]

            $data += [PSCustomObject]@{
                ID = $index
                Name = $_.Name
                "Start Frame" = $startFrame
                "End Frame" = $endFrame
                "Number of Frames" = $numberOfFrames
            }
            $index++
        }
    }

    if ($data.Count -gt 0) {
        $csvPath = Join-Path -Path $outputPath -ChildPath "animInfo.csv"
        $data | Export-Csv -Path $csvPath -NoTypeInformation
        $data | Out-GridView -Title "Frames Info"
    } else {
        Write-Host "No data collected. Please check the processing steps." -ForegroundColor Red
    }
})
$form.Controls.Add($buttonGetFramesInfo)

$form.ShowDialog() | Out-Null
