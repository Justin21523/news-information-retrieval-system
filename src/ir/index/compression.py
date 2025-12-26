"""
Index Compression

This module implements various compression techniques for inverted indexes,
reducing storage requirements while maintaining efficient decompression.

Inverted indexes can become very large, especially for web-scale collections.
Compression techniques reduce both disk space and memory usage, and can even
improve query processing speed by reducing I/O.

Key Concepts:
    - Gap Encoding: Store differences between consecutive doc IDs instead of absolute values
    - Variable-Length Encoding: Use fewer bits for small numbers
    - Prefix-Free Codes: Codes that can be uniquely decoded without delimiters

Compression Techniques:
    1. VByte (Variable Byte): Uses continuation bit in each byte
    2. Gamma Encoding: Unary length + binary offset
    3. Delta Encoding: Improved gamma with separate length encoding
    4. Rice Encoding: Parameterized encoding for specific distributions

Formulas:
    - Gap: gap_i = doc_id_i - doc_id_(i-1)
    - VByte: 7 bits data + 1 continuation bit per byte
    - Gamma: unary(⌊log2(n)⌋) + binary(n - 2^⌊log2(n)⌋)
    - Delta: gamma(⌊log2(n)⌋ + 1) + binary(n - 2^⌊log2(n)⌋)

Compression Ratios:
    - VByte: ~25-30% of uncompressed size
    - Gamma: ~15-20% of uncompressed size
    - Delta: ~10-15% of uncompressed size

Key Features:
    - Multiple compression schemes
    - Gap encoding for posting lists
    - Efficient bit-level encoding/decoding
    - Support for both doc IDs and frequencies

Reference:
    Witten, Moffat & Bell (1999). "Managing Gigabytes"
    Zobel & Moffat (2006). "Inverted Files for Text Search Engines"
    Manning et al. (2008). "Introduction to Information Retrieval" (Chapter 5)

Author: Information Retrieval System
License: Educational Use
"""

import math
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CompressionStats:
    """
    Compression statistics.

    Attributes:
        original_size: Original size in bytes
        compressed_size: Compressed size in bytes
        compression_ratio: Ratio (compressed / original)
        num_values: Number of values compressed
        encoding_type: Type of encoding used
    """
    original_size: int
    compressed_size: int
    compression_ratio: float
    num_values: int
    encoding_type: str


class VByteEncoder:
    """
    Variable Byte (VByte) encoding.

    VByte uses one or more bytes to encode an integer. Each byte has 7 bits
    for data and 1 continuation bit. If the continuation bit is 1, more bytes
    follow; if 0, this is the last byte.

    Format: [1xxxxxxx] [1xxxxxxx] ... [0xxxxxxx]
            (more)     (more)         (last)

    Complexity:
        - Encoding: O(n * log(max_value) / 7) where n = number of values
        - Decoding: O(n * log(max_value) / 7)

    Examples:
        >>> encode_vbyte(5)
        [5]  # 0000 0101

        >>> encode_vbyte(130)
        [129, 2]  # 1000 0001, 0000 0010

        >>> encode_vbyte(16384)
        [128, 128, 1]  # 1000 0000, 1000 0000, 0000 0001
    """

    def __init__(self):
        """Initialize VByte encoder."""
        self.logger = logging.getLogger(__name__)

    def encode(self, numbers: List[int]) -> bytes:
        """
        Encode list of integers using VByte encoding.

        Args:
            numbers: List of non-negative integers

        Returns:
            Encoded bytes

        Complexity:
            Time: O(n * log(max_value))
            Space: O(n * log(max_value))

        Examples:
            >>> encoder.encode([5, 130, 16384])
            b'\\x05\\x81\\x02\\x80\\x80\\x01'
        """
        result = bytearray()

        for num in numbers:
            if num < 0:
                raise ValueError(f"VByte encoding requires non-negative integers, got {num}")

            # Encode one number
            bytes_list = []
            while num >= 128:
                bytes_list.append((num % 128) | 0x80)  # Set continuation bit
                num //= 128

            bytes_list.append(num)  # Last byte (no continuation bit)
            result.extend(bytes_list)

        return bytes(result)

    def decode(self, data: bytes) -> List[int]:
        """
        Decode VByte-encoded data.

        Args:
            data: Encoded bytes

        Returns:
            List of decoded integers

        Complexity:
            Time: O(len(data))
            Space: O(number of integers)

        Examples:
            >>> encoder.decode(b'\\x05\\x81\\x02\\x80\\x80\\x01')
            [5, 130, 16384]
        """
        result = []
        current = 0
        multiplier = 1

        for byte in data:
            if byte & 0x80:  # Continuation bit set
                current += (byte & 0x7F) * multiplier
                multiplier *= 128
            else:  # Last byte
                current += byte * multiplier
                result.append(current)
                current = 0
                multiplier = 1

        return result

    def encode_gaps(self, doc_ids: List[int]) -> bytes:
        """
        Encode posting list using gap encoding + VByte.

        Args:
            doc_ids: Sorted list of document IDs

        Returns:
            Encoded bytes

        Complexity:
            Time: O(n * log(max_gap))

        Examples:
            >>> encoder.encode_gaps([3, 7, 10, 15])
            # Gaps: [3, 4, 3, 5]
            # VByte: [3, 4, 3, 5]
        """
        if not doc_ids:
            return b''

        # Compute gaps
        gaps = [doc_ids[0]]
        for i in range(1, len(doc_ids)):
            gap = doc_ids[i] - doc_ids[i - 1]
            if gap <= 0:
                raise ValueError(f"Document IDs must be sorted and unique: {doc_ids}")
            gaps.append(gap)

        return self.encode(gaps)

    def decode_gaps(self, data: bytes, length: Optional[int] = None) -> List[int]:
        """
        Decode gap-encoded posting list.

        Args:
            data: Encoded bytes
            length: Optional expected length (for validation)

        Returns:
            List of document IDs

        Complexity:
            Time: O(len(data))

        Examples:
            >>> encoder.decode_gaps(b'\\x03\\x04\\x03\\x05')
            [3, 7, 10, 15]
        """
        gaps = self.decode(data)

        if length is not None and len(gaps) != length:
            self.logger.warning(
                f"Expected {length} values, got {len(gaps)}"
            )

        # Reconstruct doc IDs from gaps
        doc_ids = []
        current_id = 0
        for gap in gaps:
            current_id += gap
            doc_ids.append(current_id)

        return doc_ids


class GammaEncoder:
    """
    Gamma (Elias Gamma) encoding.

    Gamma encoding represents an integer n ≥ 1 as:
    1. Unary encoding of ⌊log2(n)⌋ (length)
    2. Binary encoding of n - 2^⌊log2(n)⌋ (offset)

    Format: [unary(length)] [binary(offset)]

    Complexity:
        - Encoding: O(n * log(max_value))
        - Decoding: O(n * log(max_value))

    Examples:
        >>> encode_gamma(1)
        '0'  # length=0, offset=0

        >>> encode_gamma(5)
        '00101'  # length=2 (00), offset=1 (01)

        >>> encode_gamma(13)
        '0001101'  # length=3 (000), offset=5 (101)
    """

    def __init__(self):
        """Initialize Gamma encoder."""
        self.logger = logging.getLogger(__name__)

    def encode_single(self, num: int) -> str:
        """
        Encode single integer using Gamma encoding.

        Args:
            num: Positive integer (≥ 1)

        Returns:
            Binary string

        Complexity:
            Time: O(log(num))

        Examples:
            >>> encoder.encode_single(5)
            '00101'
        """
        if num < 1:
            raise ValueError(f"Gamma encoding requires positive integers, got {num}")

        # Calculate length (⌊log2(n)⌋)
        length = num.bit_length() - 1

        # Unary encoding of length (length zeros followed by 1)
        # We store it as 'length' zeros (the '1' is implicit)
        unary = '0' * length

        # Binary encoding of offset (n - 2^length)
        offset = num - (1 << length)
        binary = bin(offset)[2:].zfill(length)  # Remove '0b' prefix

        return unary + binary

    def decode_single(self, bits: str, offset: int = 0) -> Tuple[int, int]:
        """
        Decode single integer from bit string.

        Args:
            bits: Binary string
            offset: Starting position in bits

        Returns:
            (decoded_value, new_offset)

        Complexity:
            Time: O(log(value))

        Examples:
            >>> encoder.decode_single('00101', 0)
            (5, 5)
        """
        # Count leading zeros (length)
        length = 0
        pos = offset
        while pos < len(bits) and bits[pos] == '0':
            length += 1
            pos += 1

        if length == 0:
            # Special case: n = 1
            return (1, offset)

        # Read 'length' bits for offset
        if pos + length > len(bits):
            raise ValueError("Insufficient bits for Gamma decoding")

        offset_bits = bits[pos:pos + length]
        offset_value = int(offset_bits, 2) if offset_bits else 0

        # Reconstruct number
        num = (1 << length) + offset_value

        return (num, pos + length)

    def encode(self, numbers: List[int]) -> str:
        """
        Encode list of integers using Gamma encoding.

        Args:
            numbers: List of positive integers

        Returns:
            Binary string

        Complexity:
            Time: O(n * log(max_value))

        Examples:
            >>> encoder.encode([5, 13, 1])
            '001010001101110'
        """
        return ''.join(self.encode_single(num) for num in numbers)

    def decode(self, bits: str) -> List[int]:
        """
        Decode Gamma-encoded bit string.

        Args:
            bits: Binary string

        Returns:
            List of decoded integers

        Complexity:
            Time: O(len(bits))

        Examples:
            >>> encoder.decode('001010001101110')
            [5, 13, 1]
        """
        result = []
        offset = 0

        while offset < len(bits):
            num, offset = self.decode_single(bits, offset)
            result.append(num)

        return result

    def encode_gaps(self, doc_ids: List[int]) -> str:
        """
        Encode posting list using gap encoding + Gamma.

        Args:
            doc_ids: Sorted list of document IDs

        Returns:
            Binary string

        Complexity:
            Time: O(n * log(max_gap))
        """
        if not doc_ids:
            return ''

        # Compute gaps
        gaps = [doc_ids[0]]
        for i in range(1, len(doc_ids)):
            gap = doc_ids[i] - doc_ids[i - 1]
            if gap <= 0:
                raise ValueError(f"Document IDs must be sorted and unique")
            gaps.append(gap)

        return self.encode(gaps)

    def decode_gaps(self, bits: str) -> List[int]:
        """
        Decode gap-encoded posting list.

        Args:
            bits: Binary string

        Returns:
            List of document IDs

        Complexity:
            Time: O(len(bits))
        """
        gaps = self.decode(bits)

        # Reconstruct doc IDs from gaps
        doc_ids = []
        current_id = 0
        for gap in gaps:
            current_id += gap
            doc_ids.append(current_id)

        return doc_ids


class DeltaEncoder:
    """
    Delta (Elias Delta) encoding.

    Delta encoding is an improvement over Gamma encoding, using Gamma to encode
    the length, which gives better compression for larger numbers.

    Format: [gamma(length+1)] [binary(offset)]

    where length = ⌊log2(n)⌋

    Complexity:
        - Encoding: O(n * log(max_value))
        - Decoding: O(n * log(max_value))

    Examples:
        >>> encode_delta(1)
        '0'  # gamma(1)

        >>> encode_delta(5)
        '01001'  # gamma(3)=010, offset=1 (01)

        >>> encode_delta(13)
        '0100101'  # gamma(4)=00100, offset=5 (101)
    """

    def __init__(self):
        """Initialize Delta encoder."""
        self.logger = logging.getLogger(__name__)
        self.gamma_encoder = GammaEncoder()

    def encode_single(self, num: int) -> str:
        """
        Encode single integer using Delta encoding.

        Args:
            num: Positive integer (≥ 1)

        Returns:
            Binary string

        Complexity:
            Time: O(log(num))

        Examples:
            >>> encoder.encode_single(5)
            '01001'
        """
        if num < 1:
            raise ValueError(f"Delta encoding requires positive integers, got {num}")

        # Calculate length (⌊log2(n)⌋)
        length = num.bit_length() - 1

        # Gamma encode length + 1
        gamma_length = self.gamma_encoder.encode_single(length + 1)

        # Binary encoding of offset (n - 2^length)
        offset = num - (1 << length)
        binary_offset = bin(offset)[2:].zfill(length) if length > 0 else ''

        return gamma_length + binary_offset

    def decode_single(self, bits: str, offset: int = 0) -> Tuple[int, int]:
        """
        Decode single integer from bit string.

        Args:
            bits: Binary string
            offset: Starting position in bits

        Returns:
            (decoded_value, new_offset)

        Complexity:
            Time: O(log(value))
        """
        # Decode gamma-encoded length
        length_plus_1, pos = self.gamma_encoder.decode_single(bits, offset)
        length = length_plus_1 - 1

        if length == 0:
            # Special case: n = 1
            return (1, pos)

        # Read 'length' bits for offset
        if pos + length > len(bits):
            raise ValueError("Insufficient bits for Delta decoding")

        offset_bits = bits[pos:pos + length]
        offset_value = int(offset_bits, 2) if offset_bits else 0

        # Reconstruct number
        num = (1 << length) + offset_value

        return (num, pos + length)

    def encode(self, numbers: List[int]) -> str:
        """
        Encode list of integers using Delta encoding.

        Args:
            numbers: List of positive integers

        Returns:
            Binary string

        Complexity:
            Time: O(n * log(max_value))
        """
        return ''.join(self.encode_single(num) for num in numbers)

    def decode(self, bits: str) -> List[int]:
        """
        Decode Delta-encoded bit string.

        Args:
            bits: Binary string

        Returns:
            List of decoded integers

        Complexity:
            Time: O(len(bits))
        """
        result = []
        offset = 0

        while offset < len(bits):
            num, offset = self.decode_single(bits, offset)
            result.append(num)

        return result

    def encode_gaps(self, doc_ids: List[int]) -> str:
        """
        Encode posting list using gap encoding + Delta.

        Args:
            doc_ids: Sorted list of document IDs

        Returns:
            Binary string
        """
        if not doc_ids:
            return ''

        # Compute gaps
        gaps = [doc_ids[0]]
        for i in range(1, len(doc_ids)):
            gap = doc_ids[i] - doc_ids[i - 1]
            if gap <= 0:
                raise ValueError(f"Document IDs must be sorted and unique")
            gaps.append(gap)

        return self.encode(gaps)

    def decode_gaps(self, bits: str) -> List[int]:
        """
        Decode gap-encoded posting list.

        Args:
            bits: Binary string

        Returns:
            List of document IDs
        """
        gaps = self.decode(bits)

        # Reconstruct doc IDs from gaps
        doc_ids = []
        current_id = 0
        for gap in gaps:
            current_id += gap
            doc_ids.append(current_id)

        return doc_ids


def compare_compression(doc_ids: List[int]) -> dict:
    """
    Compare different compression schemes.

    Args:
        doc_ids: Sorted list of document IDs

    Returns:
        Dictionary with compression statistics for each method

    Examples:
        >>> doc_ids = [3, 7, 10, 15, 22, 30, 35]
        >>> stats = compare_compression(doc_ids)
        >>> print(stats['vbyte']['compression_ratio'])
        0.25
    """
    vbyte = VByteEncoder()
    gamma = GammaEncoder()
    delta = DeltaEncoder()

    # Original size (4 bytes per integer)
    original_size = len(doc_ids) * 4

    results = {}

    # VByte compression
    vbyte_encoded = vbyte.encode_gaps(doc_ids)
    results['vbyte'] = CompressionStats(
        original_size=original_size,
        compressed_size=len(vbyte_encoded),
        compression_ratio=len(vbyte_encoded) / original_size,
        num_values=len(doc_ids),
        encoding_type='VByte'
    )

    # Gamma compression
    gamma_encoded = gamma.encode_gaps(doc_ids)
    gamma_bytes = (len(gamma_encoded) + 7) // 8  # Convert bits to bytes
    results['gamma'] = CompressionStats(
        original_size=original_size,
        compressed_size=gamma_bytes,
        compression_ratio=gamma_bytes / original_size,
        num_values=len(doc_ids),
        encoding_type='Gamma'
    )

    # Delta compression
    delta_encoded = delta.encode_gaps(doc_ids)
    delta_bytes = (len(delta_encoded) + 7) // 8
    results['delta'] = CompressionStats(
        original_size=original_size,
        compressed_size=delta_bytes,
        compression_ratio=delta_bytes / original_size,
        num_values=len(doc_ids),
        encoding_type='Delta'
    )

    return results


def demo():
    """Demonstration of index compression."""
    print("=" * 60)
    print("Index Compression Demo")
    print("=" * 60)

    # Sample posting list
    doc_ids = [3, 7, 10, 15, 22, 30, 35, 50, 100, 200]

    print(f"\nOriginal Doc IDs: {doc_ids}")
    print(f"Original size: {len(doc_ids) * 4} bytes (4 bytes/int)")

    # VByte Encoding
    print("\n" + "-" * 60)
    print("1. VByte Encoding")
    print("-" * 60)

    vbyte = VByteEncoder()
    vbyte_encoded = vbyte.encode_gaps(doc_ids)
    vbyte_decoded = vbyte.decode_gaps(vbyte_encoded)

    print(f"Encoded size: {len(vbyte_encoded)} bytes")
    print(f"Compression ratio: {len(vbyte_encoded) / (len(doc_ids) * 4):.2%}")
    print(f"Decoded correctly: {vbyte_decoded == doc_ids}")

    # Gamma Encoding
    print("\n" + "-" * 60)
    print("2. Gamma Encoding")
    print("-" * 60)

    gamma = GammaEncoder()
    gamma_encoded = gamma.encode_gaps(doc_ids)
    gamma_decoded = gamma.decode_gaps(gamma_encoded)
    gamma_bytes = (len(gamma_encoded) + 7) // 8

    print(f"Encoded bits: {len(gamma_encoded)} bits ({gamma_bytes} bytes)")
    print(f"Compression ratio: {gamma_bytes / (len(doc_ids) * 4):.2%}")
    print(f"Decoded correctly: {gamma_decoded == doc_ids}")

    # Delta Encoding
    print("\n" + "-" * 60)
    print("3. Delta Encoding")
    print("-" * 60)

    delta = DeltaEncoder()
    delta_encoded = delta.encode_gaps(doc_ids)
    delta_decoded = delta.decode_gaps(delta_encoded)
    delta_bytes = (len(delta_encoded) + 7) // 8

    print(f"Encoded bits: {len(delta_encoded)} bits ({delta_bytes} bytes)")
    print(f"Compression ratio: {delta_bytes / (len(doc_ids) * 4):.2%}")
    print(f"Decoded correctly: {delta_decoded == doc_ids}")

    # Comparison
    print("\n" + "-" * 60)
    print("Compression Comparison")
    print("-" * 60)

    stats = compare_compression(doc_ids)
    print(f"\n{'Method':<15} {'Size (bytes)':<15} {'Ratio':<10} {'Savings'}")
    print("-" * 60)

    for method, stat in stats.items():
        savings = (1 - stat.compression_ratio) * 100
        print(f"{method.upper():<15} {stat.compressed_size:<15} "
              f"{stat.compression_ratio:<10.2%} {savings:>6.1f}%")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
