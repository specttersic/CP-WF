@echo off
chcp 65001 >nul
REM ============================================================
REM 消融研究 - Mockingbird 防御，子数据集 A，20 epochs
REM
REM 完整模型直接使用现有结果：
REM   saved_models/splited/denoiser_mockingbird_datasetA.pth
REM
REM 消融变体（重新训练）：
REM   wo_per    - 移除感知损失 (L_rec + L_con)
REM   wo_con    - 移除对比损失 (L_rec + L_per)
REM   only_rec  - 仅重建损失  (L_rec)
REM   full+no_ag- 移除 AG
REM ============================================================

set DEFENSE=mockingbird
set DATASET=A
set EPOCHS=20
set BATCH=32

call conda activate pytorch

cd /d "%~dp0..\.."

echo ============================================================
echo 消融研究  防御: %DEFENSE%  数据集: %DATASET%  Epochs: %EPOCHS%
echo ============================================================
echo.

REM ── 1. w/o L_per ────────────────────────────────────────────
echo [1/4] 训练 w/o L_per...
python improved_denoiser/ablation/train_ablation.py ^
    --ablation wo_per --defense %DEFENSE% ^
    --datasets %DATASET% --epochs %EPOCHS% --batch_size %BATCH% ^
    --data_path processed_data ^
    --encoder_path improved_denoiser/saved_models/pretrained_encoder.pth ^
    --save_dir improved_denoiser/saved_models/ablation ^
    --result_dir improved_denoiser/results/ablation
if %errorlevel% neq 0 ( echo 错误: wo_per 训练失败 & pause & exit /b 1 )
echo.

REM ── 2. w/o L_con ────────────────────────────────────────────
echo [2/4] 训练 w/o L_con...
python improved_denoiser/ablation/train_ablation.py ^
    --ablation wo_con --defense %DEFENSE% ^
    --datasets %DATASET% --epochs %EPOCHS% --batch_size %BATCH% ^
    --data_path processed_data ^
    --encoder_path improved_denoiser/saved_models/pretrained_encoder.pth ^
    --save_dir improved_denoiser/saved_models/ablation ^
    --result_dir improved_denoiser/results/ablation
if %errorlevel% neq 0 ( echo 错误: wo_con 训练失败 & pause & exit /b 1 )
echo.

REM ── 3. 仅 L_rec ─────────────────────────────────────────────
echo [3/4] 训练 only_rec...
python improved_denoiser/ablation/train_ablation.py ^
    --ablation only_rec --defense %DEFENSE% ^
    --datasets %DATASET% --epochs %EPOCHS% --batch_size %BATCH% ^
    --data_path processed_data ^
    --encoder_path improved_denoiser/saved_models/pretrained_encoder.pth ^
    --save_dir improved_denoiser/saved_models/ablation ^
    --result_dir improved_denoiser/results/ablation
if %errorlevel% neq 0 ( echo 错误: only_rec 训练失败 & pause & exit /b 1 )
echo.

REM ── 4. w/o AG ───────────────────────────────────────────────
echo [4/4] 训练 w/o AG...
python improved_denoiser/ablation/train_ablation.py ^
    --ablation full --no_ag --defense %DEFENSE% ^
    --datasets %DATASET% --epochs %EPOCHS% --batch_size %BATCH% ^
    --data_path processed_data ^
    --encoder_path improved_denoiser/saved_models/pretrained_encoder.pth ^
    --save_dir improved_denoiser/saved_models/ablation ^
    --result_dir improved_denoiser/results/ablation
if %errorlevel% neq 0 ( echo 错误: wo_ag 训练失败 & pause & exit /b 1 )
echo.

REM ── 评估：消融变体 + 现有完整模型 ───────────────────────────
echo ============================================================
echo 评估所有消融变体...
echo ============================================================
python improved_denoiser/ablation/evaluate_ablation_mockingbird.py ^
    --defense %DEFENSE% --dataset %DATASET% ^
    --data_path processed_data ^
    --full_model_path improved_denoiser/saved_models/splited/denoiser_mockingbird_datasetA.pth ^
    --ablation_model_dir improved_denoiser/saved_models/ablation ^
    --classifier_dir saved_models ^
    --result_dir improved_denoiser/results/ablation
if %errorlevel% neq 0 ( echo 错误: 评估失败 & pause & exit /b 1 )

cd improved_denoiser\ablation

echo.
echo ============================================================
echo 消融研究完成！结果保存在 improved_denoiser/results/ablation/
echo ============================================================
pause
