
import argparse
import asyncio
from metetl.logging_config import get_logger
from metetl.images.processing import prepare_json, ImageProcessor
from metetl.analysis.aggregations import run_analysis_pipeline, draw_results

logger = get_logger()

def create_parser():
    parser = argparse.ArgumentParser(description="MetETL: Инструмент для скачивания, обработки и анализа данных.")
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    prep_parser = subparsers.add_parser("prepare", help="Подготовка json файла с метаданными")
    prep_parser.add_argument("--csv", required=True, help="Путь к исходному CSV файлу")
    prep_parser.add_argument("--output", required=True, help="Путь для сохранения JSON файла")

    proc_parser = subparsers.add_parser("process", help="Запуск пайплайна скачивания и обработки")
    proc_parser.add_argument("--input", required=True, help="Путь к JSON файлу")
    proc_parser.add_argument("--output", required=True, help="Директория для сохранения изображений")
    proc_parser.add_argument("--num", type=int, required=True, help="Количество изображений (Обязательно)") # [cite: 24]

    ana_parser = subparsers.add_parser("analyze", help="Запуск анализа датасета")
    ana_parser.add_argument("--csv", required=True, help="Путь к CSV файлу")
    ana_parser.add_argument("--output-dir", required=True, help="Директория для графиков")

    return parser

def run_cli():
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    logger.info("Старт программы")

    try:
        if args.command == "prepare":
            prepare_json(args.csv, args.output)
            
        elif args.command == "process":
            processor = ImageProcessor(output_dir=args.output)
            asyncio.run(processor.run(args.input, args.num))
            
        elif args.command == "analyze":
            df = run_analysis_pipeline(args.csv)
            draw_results(df, args.output_dir)
            
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}", exc_info=True)
    
    logger.info("Завершение программы")