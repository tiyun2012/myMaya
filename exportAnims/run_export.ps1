
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# Define the path to mayapy.exe (update if necessary)
$MAYA_PY = "C:\Program Files\Autodesk\Maya2024\bin\mayapy.exe"

# Load required .NET assemblies for Windows Forms and Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

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

# Create main form
$form = New-Object System.Windows.Forms.Form
$form.Text = "Path Selector"
$form.Size = New-Object System.Drawing.Size(420, 300)
$form.StartPosition = 'CenterScreen'

# Create exportGroup (GroupBox) to hold sourceGroup, outGroup and Export button
$exportGroup = New-Object System.Windows.Forms.GroupBox
$exportGroup.Text = "Export Options"
$exportGroup.Size = New-Object System.Drawing.Size(380, 170)
$exportGroup.Location = New-Object System.Drawing.Point(10, 10)

# Create sourceGroup inside exportGroup
$sourceGroup = New-Object System.Windows.Forms.GroupBox
$sourceGroup.Text = "Source Path"
$sourceGroup.Size = New-Object System.Drawing.Size(360, 50)
$sourceGroup.Location = New-Object System.Drawing.Point(10, 20)

$labelSource = New-Object System.Windows.Forms.Label
$labelSource.Text = "Source Path:"
$labelSource.Location = New-Object System.Drawing.Point(10, 20)
$labelSource.AutoSize = $true
$labelSource.ForeColor = 'Blue'
$labelSource.Cursor = [System.Windows.Forms.Cursors]::Hand
$labelSource.Add_Click({
    $path = Select-FolderDialog
    if ($path) { $textSource.Text = $path }
})

$textSource = New-Object System.Windows.Forms.TextBox
$textSource.Location = New-Object System.Drawing.Point(100, 18)
$textSource.Size = New-Object System.Drawing.Size(200, 20)

$buttonBrowseSource = New-Object System.Windows.Forms.Button
$buttonBrowseSource.Text = "..."
$buttonBrowseSource.Location = New-Object System.Drawing.Point(310, 16)
$buttonBrowseSource.Size = New-Object System.Drawing.Size(30, 23)
$buttonBrowseSource.Add_Click({
    $path = Select-FolderDialog
    if ($path) { $textSource.Text = $path }
})

$sourceGroup.Controls.Add($labelSource)
$sourceGroup.Controls.Add($textSource)
$sourceGroup.Controls.Add($buttonBrowseSource)

# Create outGroup inside exportGroup
$outGroup = New-Object System.Windows.Forms.GroupBox
$outGroup.Text = "Output Path"
$outGroup.Size = New-Object System.Drawing.Size(360, 50)
$outGroup.Location = New-Object System.Drawing.Point(10, 80)

$labelOutput = New-Object System.Windows.Forms.Label
$labelOutput.Text = "Output Path:"
$labelOutput.Location = New-Object System.Drawing.Point(10, 20)
$labelOutput.AutoSize = $true
$labelOutput.ForeColor = 'Blue'
$labelOutput.Cursor = [System.Windows.Forms.Cursors]::Hand
$labelOutput.Add_Click({
    $path = Select-FolderDialog
    if ($path) { $textOutput.Text = $path }
})

$textOutput = New-Object System.Windows.Forms.TextBox
$textOutput.Location = New-Object System.Drawing.Point(100, 18)
$textOutput.Size = New-Object System.Drawing.Size(200, 20)

$buttonBrowseOutput = New-Object System.Windows.Forms.Button
$buttonBrowseOutput.Text = "..."
$buttonBrowseOutput.Location = New-Object System.Drawing.Point(310, 16)
$buttonBrowseOutput.Size = New-Object System.Drawing.Size(30, 23)
$buttonBrowseOutput.Add_Click({
    $path = Select-FolderDialog
    if ($path) { $textOutput.Text = $path }
})

$outGroup.Controls.Add($labelOutput)
$outGroup.Controls.Add($textOutput)
$outGroup.Controls.Add($buttonBrowseOutput)

# Create Export button inside exportGroup
$buttonExport = New-Object System.Windows.Forms.Button
$buttonExport.Text = "Export"
$buttonExport.Location = New-Object System.Drawing.Point(140, 135)
$buttonExport.Size = New-Object System.Drawing.Size(100, 25)
$buttonExport.Add_Click({
    $sourcePath = $textSource.Text
    $outputPath = $textOutput.Text

    if (-not (Test-Path -Path $sourcePath)) {
        [System.Windows.Forms.MessageBox]::Show("Source path does not exist!", "Error")
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

    [System.Windows.Forms.MessageBox]::Show("Exporting completed!", "Export")
})

$exportGroup.Controls.Add($sourceGroup)
$exportGroup.Controls.Add($outGroup)
$exportGroup.Controls.Add($buttonExport)

# Create frameinfoGroup to hold Frames Info controls
$frameinfoGroup = New-Object System.Windows.Forms.GroupBox
$frameinfoGroup.Text = "Frame Info Options"
$frameinfoGroup.Size = New-Object System.Drawing.Size(380, 70)
$frameinfoGroup.Location = New-Object System.Drawing.Point(10, 180)

$labelCsvName = New-Object System.Windows.Forms.Label
$labelCsvName.Text = "CSV Name:"
$labelCsvName.Location = New-Object System.Drawing.Point(10, 30)
$labelCsvName.AutoSize = $true

$csvNameTextBox = New-Object System.Windows.Forms.TextBox
$csvNameTextBox.Location = New-Object System.Drawing.Point(80, 28)
$csvNameTextBox.Size = New-Object System.Drawing.Size(200, 20)
# Set a default CSV name
$csvNameTextBox.Text = "animInfo.csv"

$buttonGetFramesInfo = New-Object System.Windows.Forms.Button
$buttonGetFramesInfo.Text = "Get Frames"
$buttonGetFramesInfo.Location = New-Object System.Drawing.Point(290, 26)
$buttonGetFramesInfo.Size = New-Object System.Drawing.Size(80, 25)
$buttonGetFramesInfo.Add_Click({
    $sourcePath = $textSource.Text
    $outputPath = $textOutput.Text
    $csvName = $csvNameTextBox.Text

    if (-not (Test-Path -Path $sourcePath)) {
        [System.Windows.Forms.MessageBox]::Show("Source path does not exist!", "Error")
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
        $csvPath = Join-Path -Path $outputPath -ChildPath $csvName
        $data | Export-Csv -Path $csvPath -NoTypeInformation
        $data | Out-GridView -Title "Frames Info"
    } else {
        Write-Host "No data collected. Please check the processing steps." -ForegroundColor Red
    }
})

$frameinfoGroup.Controls.Add($labelCsvName)
$frameinfoGroup.Controls.Add($csvNameTextBox)
$frameinfoGroup.Controls.Add($buttonGetFramesInfo)

$form.Controls.Add($exportGroup)
$form.Controls.Add($frameinfoGroup)

$form.ShowDialog() | Out-Null
