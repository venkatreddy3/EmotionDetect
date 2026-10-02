# Execute CNN Training
param(
    [string]$DataDir = "data/raw/train",
    [string]$ModelType = "mini_xception",
    [int]$Epochs = 50,
    [int]$BatchSize = 64
)

Write-Host "Launching CNN training ($ModelType) for $Epochs epochs..." -ForegroundColor Cyan
python -m training.train --data-dir $DataDir --model-type $ModelType --epochs $Epochs --batch-size $BatchSize
