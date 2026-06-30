@echo off

if "%~1"=="" (
    echo Usage: indexer.bat ^<input-dir^> ^<output-dir^>
    echo Example: indexer.bat ..\data\crawled ..\data\msmarco_index
    exit /b 1
)

if "%~2"=="" (
    echo Usage: indexer.bat ^<input-dir^> ^<output-dir^>
    echo Example: indexer.bat ..\data\crawled ..\data\msmarco_index
    exit /b 1
)

python bert_indexer.py --input_dir "%~1" --output_dir "%~2"
