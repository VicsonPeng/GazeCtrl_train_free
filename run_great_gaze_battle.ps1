$warp_modes = "none", "shift", "tps", "blur"
$prompt_styles = "standard", "contrastive", "vector_focus"
$vector_options = $false, $true

if (!(Test-Path results_ab_test)) { New-Item -ItemType Directory -Path results_ab_test }

foreach ($w in $warp_modes) {
    foreach ($p in $prompt_styles) {
        foreach ($v in $vector_options) {
            $v_flag = if ($v) { "--use-vector" } else { "" }
            $v_str = if ($v) { "vector" } else { "novector" }
            $out_dir = "results_ab_test/res_${w}_${v_str}_${p}"
            
            Write-Host "`n>>> Running: Warp=$w, Vector=$v_str, Prompt=$p" -ForegroundColor Cyan
            python batch_evaluate.py `
                --input-dir ../L2CS-Net/datasets/SHHQ-1.0/no_segment `
                --max-images 10 `
                --warp-mode $w `
                $v_flag `
                --prompt-style $p `
                --output-dir $out_dir `
                --seed 42
        }
    }
}
