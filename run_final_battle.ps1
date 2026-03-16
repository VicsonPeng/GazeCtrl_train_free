# run_final_battle.ps1
# Full experiment sweep: 3 warp x 2 vector x 5 prompts = 30 combinations (all two-pass)
# 30 images x 30 combos = 900 total evaluations

$inputDir = "../L2CS-Net/datasets/SHHQ-1.0/no_segment"
$baseOut  = "results_final"
$seed     = 42
$maxImg   = 15

$warps   = @("none", "shift", "tps")
$vectors = @($false, $true)
$prompts = @("standard", "contrastive", "vector_focus", "p1_gaze_only", "p1_contrastive_gaze")

$total = $warps.Count * $vectors.Count * $prompts.Count
$i = 0

foreach ($w in $warps) {
    foreach ($v in $vectors) {
        foreach ($p in $prompts) {
            $i++
            $vTag = if ($v) { "vector" } else { "novector" }
            $outDir = "$baseOut/res_${w}_${vTag}_${p}"

            Write-Host ""
            Write-Host "============================================================"
            Write-Host "  [$i/$total] warp=$w  vector=$vTag  prompt=$p"
            Write-Host "  Output: $outDir"
            Write-Host "============================================================"

            $args = @(
                "batch_evaluate.py",
                "--input-dir", $inputDir,
                "--output-dir", $outDir,
                "--warp-mode", $w,
                "--prompt-style", $p,
                "--max-images", $maxImg,
                "--seed", $seed
            )
            if ($v) { $args += "--use-vector" }

            python @args
        }
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "  ALL $total COMBINATIONS COMPLETE"
Write-Host "  Running aggregation..."
Write-Host "============================================================"

python aggregate_results.py --results-dir $baseOut
