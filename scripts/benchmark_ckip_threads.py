#!/usr/bin/env python
"""
CKIP Tokenizer Threading Benchmark

Benchmark CKIP tokenization performance with different thread counts.
Helps determine optimal threading configuration for your system.

Usage:
    python scripts/benchmark_ckip_threads.py
    python scripts/benchmark_ckip_threads.py --threads 16 32
    python scripts/benchmark_ckip_threads.py --sentences 100

Author: Information Retrieval System
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.text.ckip_tokenizer_optimized import CKIPTokenizerOptimized


def generate_test_sentences(count: int = 100) -> List[str]:
    """
    Generate test sentences for benchmarking.

    Args:
        count: Number of sentences to generate

    Returns:
        List of test sentences
    """
    base_sentences = [
        "台灣是一個美麗的島嶼，擁有豐富的自然資源和多元的文化。",
        "人工智慧技術在近年來取得了突破性的進展，深度學習模型越來越強大。",
        "新聞報導指出，今年經濟成長率達到了預期目標，產業復甦態勢良好。",
        "氣候變遷問題日益嚴重，各國政府必須採取更積極的行動來減少碳排放。",
        "科技公司持續投資研發，希望能夠開發出更創新的產品和服務。",
        "教育改革是社會進步的關鍵，需要政府、學校和家長共同努力。",
        "健康醫療體系面臨諸多挑戰，包括人口老化和醫療資源分配不均。",
        "金融市場波動加劇，投資人需要更謹慎地評估風險和機會。",
        "環保意識抬頭，越來越多企業開始重視永續發展和社會責任。",
        "數位轉型已成為企業生存的必要條件，傳統產業也需要積極適應。",
        "都市化進程加速，城市規劃必須兼顧經濟發展和生活品質。",
        "網路安全威脅持續增加，個人和企業都需要加強資訊保護措施。",
        "再生能源產業快速發展，太陽能和風力發電成本持續降低。",
        "人才培育是國家競爭力的根本，教育投資不容忽視。",
        "疫情改變了工作型態，遠距辦公成為新常態。",
    ]

    # Repeat to reach desired count
    sentences = []
    while len(sentences) < count:
        sentences.extend(base_sentences)

    return sentences[:count]


def benchmark_threading(thread_counts: List[int],
                        num_sentences: int = 100,
                        batch_size: int = 512) -> dict:
    """
    Benchmark CKIP tokenization with different thread counts.

    Args:
        thread_counts: List of thread counts to test
        num_sentences: Number of sentences to process
        batch_size: Batch size for tokenization

    Returns:
        Dictionary with benchmark results
    """
    logger = logging.getLogger(__name__)

    # Generate test data
    logger.info(f"Generating {num_sentences} test sentences...")
    test_sentences = generate_test_sentences(num_sentences)

    results = {}

    for threads in thread_counts:
        logger.info(f"\n{'='*80}")
        logger.info(f"Benchmarking with {threads} threads")
        logger.info(f"{'='*80}")

        # Reset singleton for each test
        CKIPTokenizerOptimized._instance = None
        CKIPTokenizerOptimized._initialized = False

        try:
            # Initialize tokenizer
            logger.info("Initializing CKIP tokenizer...")
            init_start = time.time()
            tokenizer = CKIPTokenizerOptimized(num_threads=threads)
            init_time = time.time() - init_start
            logger.info(f"  Initialization: {init_time:.4f}s")

            # Warm-up run
            logger.info("Warm-up run...")
            _ = tokenizer.tokenize_batch(test_sentences[:10], batch_size=batch_size)

            # Benchmark run
            logger.info(f"Processing {num_sentences} sentences (batch_size={batch_size})...")
            start_time = time.time()
            token_results = tokenizer.tokenize_batch(test_sentences, batch_size=batch_size)
            elapsed = time.time() - start_time

            # Calculate statistics
            total_tokens = sum(len(tokens) for tokens in token_results)
            throughput = num_sentences / elapsed
            tokens_per_sec = total_tokens / elapsed

            results[threads] = {
                'threads': threads,
                'sentences': num_sentences,
                'batch_size': batch_size,
                'init_time': init_time,
                'processing_time': elapsed,
                'total_tokens': total_tokens,
                'throughput': throughput,  # sentences/sec
                'tokens_per_sec': tokens_per_sec,
                'avg_time_per_sentence': elapsed / num_sentences * 1000,  # ms
            }

            logger.info(f"\nResults:")
            logger.info(f"  Processing time: {elapsed:.4f}s")
            logger.info(f"  Throughput: {throughput:.2f} sentences/sec")
            logger.info(f"  Token throughput: {tokens_per_sec:.2f} tokens/sec")
            logger.info(f"  Total tokens: {total_tokens:,}")
            logger.info(f"  Avg per sentence: {elapsed / num_sentences * 1000:.2f}ms")

            # Get tokenizer stats
            stats = tokenizer.get_stats()
            logger.info(f"\nTokenizer configuration:")
            for k, v in stats.items():
                logger.info(f"  {k}: {v}")

        except Exception as e:
            logger.error(f"Error with {threads} threads: {e}", exc_info=True)
            results[threads] = {'error': str(e)}

    return results


def print_comparison(results: dict):
    """
    Print comparison table of benchmark results.

    Args:
        results: Dictionary with benchmark results
    """
    print("\n" + "="*100)
    print("CKIP TOKENIZATION PERFORMANCE COMPARISON")
    print("="*100)
    print()

    # Table header
    header = f"{'Threads':<10} {'Time (s)':<12} {'Throughput':<20} {'Tokens/sec':<18} {'Speedup':<12}"
    print(header)
    print("-" * 100)

    # Get baseline (first result)
    baseline_threads = min(results.keys())
    baseline_time = results[baseline_threads]['processing_time']

    # Print results
    for threads in sorted(results.keys()):
        r = results[threads]

        if 'error' in r:
            print(f"{threads:<10} ERROR: {r['error']}")
            continue

        time_val = r['processing_time']
        throughput = r['throughput']
        tokens_per_sec = r['tokens_per_sec']
        speedup = baseline_time / time_val

        print(f"{threads:<10} {time_val:<12.4f} {throughput:<20.2f} {tokens_per_sec:<18.2f} {speedup:<12.2f}x")

    print("-" * 100)

    # Find best configuration
    best_threads = max(results.keys(), key=lambda k: results[k].get('throughput', 0))
    best_throughput = results[best_threads]['throughput']
    best_speedup = baseline_time / results[best_threads]['processing_time']

    print(f"\n🏆 Best configuration: {best_threads} threads")
    print(f"   Throughput: {best_throughput:.2f} sentences/sec")
    print(f"   Speedup: {best_speedup:.2f}x over baseline ({baseline_threads} threads)")
    print("="*100 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Benchmark CKIP tokenization with different thread counts'
    )
    parser.add_argument(
        '--threads',
        type=int,
        nargs='+',
        default=[8, 16, 24, 32],
        help='Thread counts to test (default: 8 16 24 32)'
    )
    parser.add_argument(
        '--sentences',
        type=int,
        default=100,
        help='Number of sentences to process (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size for tokenization (default: 512)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output (DEBUG level)'
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("CKIP Tokenizer Threading Benchmark")
    logger.info("="*80)
    logger.info(f"Test configuration:")
    logger.info(f"  Thread counts: {args.threads}")
    logger.info(f"  Sentences: {args.sentences}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info("="*80)

    # Run benchmark
    results = benchmark_threading(
        thread_counts=args.threads,
        num_sentences=args.sentences,
        batch_size=args.batch_size
    )

    # Print comparison
    print_comparison(results)

    # Recommendation
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"💡 System information:")
    print(f"   Available CPU threads: {cpu_count}")
    print(f"\n💡 Recommendation:")
    print(f"   For maximum speed, use: --num-threads {cpu_count}")
    print(f"   For balanced performance: --num-threads {cpu_count // 2}")
    print()


if __name__ == '__main__':
    main()
