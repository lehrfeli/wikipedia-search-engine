@echo off
:: Check if all 4 arguments are provided (%4 refers to the fourth argument)
if "%~4"=="" (
    echo Usage: crawler.bat ^<seed-file^> ^<num-pages^> ^<hops-away^> ^<output-dir^>
    exit /b 1
)

:: Assign variables for clarity (optional, but helpful)
set SEED_FILE=%1
set PAGES=%2
set HOPS=%3
set OUT_DIR=%4

echo Launching crawler with seed: %SEED_FILE%, Limit: %PAGES% pages, Depth: %HOPS% hops...

:: Run Scrapy and pass the batch arguments into the spider
scrapy runspider wiki_spider.py -a seed_file="%SEED_FILE%" -a num_pages="%PAGES%" -a hops="%HOPS%" -a output_dir="%OUT_DIR%"