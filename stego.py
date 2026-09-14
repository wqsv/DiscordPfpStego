#!/usr/bin/env python3
"""
Keyless robust DCT steganography for short text through lossy image conversion.

Encoding:
    python stego.py encode source.png encoded.png "Hello from Discord."

Decoding:
    python stego.py decode downloaded.webp

Design:
    - JPEG-style 8x8 DCT
    - Coefficient-difference embedding
    - BCH(31,21), correcting up to 2 bit errors per codeword
    - Deterministic interleaving
    - Repetition coding
    - No password or secret key
"""

from __future__ import annotations

import argparse
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BLOCK_SIZE = 8
TARGET_SIZE = (1024, 1024)

DELTA = 25.0

HEADER_REPETITION = 25
PAYLOAD_REPETITION = 30

MAGIC = b"BCH7"
MAX_MESSAGE_BYTES = 36

SHUFFLE_SEED = 0x6D2B79F5

HEADER_FORMAT = ">4sHI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


# Low-frequency AC coefficient pairs.
# The same bit is embedded into every pair in a block.
COEFFICIENT_PAIRS = (
    ((2, 3), (3, 2)),
)


# BCH(31,21), t=2
BCH_N = 31
BCH_K = 21
BCH_R = 10
BCH_GENERATOR = 0x769


# ---------------------------------------------------------------------------
# DCT
# ---------------------------------------------------------------------------

def build_dct_matrix() -> np.ndarray:
    """Build the 8x8 orthonormal DCT transform matrix."""
    matrix = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)

    for u in range(BLOCK_SIZE):
        scale = math.sqrt(1 / BLOCK_SIZE) if u == 0 else math.sqrt(2 / BLOCK_SIZE)

        for x in range(BLOCK_SIZE):
            matrix[u, x] = (
                scale
                * math.cos(math.pi * (2 * x + 1) * u / (2 * BLOCK_SIZE))
            )

    return matrix


DCT_MATRIX = build_dct_matrix()


def dct8(block: np.ndarray) -> np.ndarray:
    """Apply an 8x8 2D DCT."""
    return DCT_MATRIX @ block @ DCT_MATRIX.T


def idct8(coefficients: np.ndarray) -> np.ndarray:
    """Apply an 8x8 inverse 2D DCT."""
    return DCT_MATRIX.T @ coefficients @ DCT_MATRIX


# ---------------------------------------------------------------------------
# BCH
# ---------------------------------------------------------------------------

def polynomial_mod(value: int) -> int:
    """Return value modulo the BCH generator polynomial."""
    while value and value.bit_length() - 1 >= BCH_R:
        shift = (value.bit_length() - 1) - BCH_R
        value ^= BCH_GENERATOR << shift

    return value


def encode_bch_word(data: int) -> int:
    """Encode one 21-bit value into a 31-bit BCH codeword."""
    data &= (1 << BCH_K) - 1

    shifted = data << BCH_R
    remainder = polynomial_mod(shifted)

    return shifted | remainder


def build_error_lookup() -> dict[int, int]:
    """
    Build a syndrome -> error-pattern lookup for all errors of weight <= 2.
    """
    lookup = {0: 0}

    # One-bit errors.
    for i in range(BCH_N):
        error = 1 << i
        syndrome = polynomial_mod(error)

        if syndrome in lookup and lookup[syndrome] != error:
            raise RuntimeError("Invalid BCH generator: syndrome collision")

        lookup[syndrome] = error

    # Two-bit errors.
    for i in range(BCH_N):
        for j in range(i + 1, BCH_N):
            error = (1 << i) | (1 << j)
            syndrome = polynomial_mod(error)

            if syndrome in lookup and lookup[syndrome] != error:
                raise RuntimeError("Invalid BCH generator: syndrome collision")

            lookup[syndrome] = error

    return lookup


ERROR_BY_SYNDROME = build_error_lookup()


def bch_encode(bits: list[int]) -> list[int]:
    """Encode an arbitrary bit stream using BCH(31,21)."""
    encoded: list[int] = []

    for start in range(0, len(bits), BCH_K):
        chunk = bits[start:start + BCH_K]
        chunk += [0] * (BCH_K - len(chunk))

        data = 0
        for bit in chunk:
            data = (data << 1) | int(bit)

        codeword = encode_bch_word(data)

        encoded.extend(
            (codeword >> (BCH_N - 1 - i)) & 1
            for i in range(BCH_N)
        )

    return encoded


@dataclass
class BCHDecodeResult:
    """Statistics and recovered bits from BCH decoding."""

    bits: list[int]
    corrected_codewords: int
    uncorrectable_codewords: int
    codewords: int


def bch_decode(bits: list[int], raw_bit_count: int) -> BCHDecodeResult:
    """Decode BCH(31,21) codewords and correct up to two errors each."""
    output: list[int] = []

    corrected = 0
    uncorrectable = 0
    codeword_count = 0

    for start in range(0, len(bits), BCH_N):
        chunk = bits[start:start + BCH_N]

        if len(chunk) < BCH_N:
            break

        codeword_count += 1

        word = 0
        for bit in chunk:
            word = (word << 1) | int(bit)

        syndrome = polynomial_mod(word)

        if syndrome:
            error = ERROR_BY_SYNDROME.get(syndrome)

            if error is None:
                uncorrectable += 1
                continue

            word ^= error
            corrected += 1

        # Defensive validation after correction.
        if polynomial_mod(word) != 0:
            uncorrectable += 1
            continue

        data = word >> BCH_R

        output.extend(
            (data >> (BCH_K - 1 - i)) & 1
            for i in range(BCH_K)
        )

    return BCHDecodeResult(
        bits=output[:raw_bit_count],
        corrected_codewords=corrected,
        uncorrectable_codewords=uncorrectable,
        codewords=codeword_count,
    )


# ---------------------------------------------------------------------------
# Bit / byte helpers
# ---------------------------------------------------------------------------

def bytes_to_bits(data: bytes) -> list[int]:
    """Convert bytes to a big-endian bit list."""
    return [
        (byte >> (7 - bit_index)) & 1
        for byte in data
        for bit_index in range(8)
    ]


def bits_to_bytes(bits: list[int]) -> bytes:
    """Convert a bit list to bytes."""
    output = bytearray()

    for start in range(0, len(bits) - 7, 8):
        value = 0

        for bit in bits[start:start + 8]:
            value = (value << 1) | int(bit)

        output.append(value)

    return bytes(output)


def coded_length(raw_bits: int) -> int:
    """Return the number of BCH-coded bits required for raw_bit_count bits."""
    codewords = (raw_bits + BCH_K - 1) // BCH_K
    return codewords * BCH_N


# ---------------------------------------------------------------------------
# Deterministic shuffling / interleaving
# ---------------------------------------------------------------------------

def permutation(size: int) -> list[int]:
    """Return a deterministic Fisher-Yates permutation."""
    result = list(range(size))
    state = SHUFFLE_SEED & 0xFFFFFFFF

    for index in range(size - 1, 0, -1):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        swap_index = state % (index + 1)

        result[index], result[swap_index] = (
            result[swap_index],
            result[index],
        )

    return result


def interleave(bits: list[int]) -> list[int]:
    """Apply the deterministic bit interleaver."""
    order = permutation(len(bits))
    output = [0] * len(bits)

    for index, bit in enumerate(bits):
        output[order[index]] = int(bit)

    return output


def deinterleave(bits: list[int]) -> list[int]:
    """Reverse the deterministic bit interleaving."""
    order = permutation(len(bits))
    output = [0] * len(bits)

    for index in range(len(bits)):
        output[index] = int(bits[order[index]])

    return output


def block_order(width: int, height: int) -> list[tuple[int, int]]:
    """Return image blocks in deterministic shuffled order."""
    blocks = [
        (y, x)
        for y in range(0, height - BLOCK_SIZE + 1, BLOCK_SIZE)
        for x in range(0, width - BLOCK_SIZE + 1, BLOCK_SIZE)
    ]

    state = SHUFFLE_SEED & 0xFFFFFFFF

    for index in range(len(blocks) - 1, 0, -1):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        swap_index = state % (index + 1)

        blocks[index], blocks[swap_index] = (
            blocks[swap_index],
            blocks[index],
        )

    return blocks


def repeat_bits(bits: list[int], count: int) -> list[int]:
    """Repeat every bit count times."""
    output: list[int] = []

    for bit in bits:
        output.extend([int(bit)] * count)

    return output


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def make_payload(message: str) -> tuple[bytes, bytes]:
    """
    Build the header and UTF-8 payload.

    Header:
        magic       4 bytes
        message_len 2 bytes
        CRC32       4 bytes
    """
    data = message.encode("utf-8")

    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Message is {len(data)} UTF-8 bytes; "
            f"maximum is {MAX_MESSAGE_BYTES} bytes."
        )

    checksum = zlib.crc32(data) & 0xFFFFFFFF
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        len(data),
        checksum,
    )

    return header, data


def capacity(width: int, height: int) -> int:
    """Estimate the maximum UTF-8 payload size in bytes."""
    block_count = (width // BLOCK_SIZE) * (height // BLOCK_SIZE)

    header_bits = HEADER_SIZE * 8
    header_coded_bits = coded_length(header_bits)
    header_blocks = header_coded_bits * HEADER_REPETITION

    if block_count <= header_blocks:
        return 0

    remaining_blocks = block_count - header_blocks
    max_coded_bits = remaining_blocks // PAYLOAD_REPETITION
    max_codewords = max_coded_bits // BCH_N

    return min(
        MAX_MESSAGE_BYTES,
        (max_codewords * BCH_K) // 8,
    )


# ---------------------------------------------------------------------------
# DCT embedding / extraction
# ---------------------------------------------------------------------------

def set_pair_bit(
    coefficients: np.ndarray,
    pair: tuple[tuple[int, int], tuple[int, int]],
    bit: int,
) -> None:
    """Force the coefficient difference to represent one bit."""
    first, second = pair

    a = float(coefficients[first])
    b = float(coefficients[second])

    midpoint = (a + b) / 2.0
    target_difference = DELTA if bit else -DELTA

    coefficients[first] = midpoint + target_difference / 2.0
    coefficients[second] = midpoint - target_difference / 2.0


def read_pair(
    coefficients: np.ndarray,
    pair: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, float]:
    """Read one bit and its coefficient-difference confidence."""
    first, second = pair
    difference = float(coefficients[first] - coefficients[second])

    bit = 1 if difference >= 0 else 0
    confidence = abs(difference)

    return bit, confidence


def embed_block(
    luminance: np.ndarray,
    y: int,
    x: int,
    bit: int,
) -> None:
    """Embed one bit into one 8x8 luminance block."""
    block = luminance[
        y:y + BLOCK_SIZE,
        x:x + BLOCK_SIZE,
    ]

    coefficients = dct8(block)

    for pair in COEFFICIENT_PAIRS:
        set_pair_bit(coefficients, pair, bit)

    encoded_block = np.rint(idct8(coefficients))
    luminance[
        y:y + BLOCK_SIZE,
        x:x + BLOCK_SIZE,
    ] = np.clip(encoded_block, 0, 255)


def read_block(
    luminance: np.ndarray,
    y: int,
    x: int,
) -> tuple[int, float]:
    """Read one bit from one 8x8 luminance block."""
    block = luminance[
        y:y + BLOCK_SIZE,
        x:x + BLOCK_SIZE,
    ]

    coefficients = dct8(block)

    votes = 0
    confidence = 0.0

    for pair in COEFFICIENT_PAIRS:
        bit, pair_confidence = read_pair(coefficients, pair)
        votes += bit
        confidence += pair_confidence

    return (
        1 if votes >= len(COEFFICIENT_PAIRS) / 2 else 0,
        confidence / len(COEFFICIENT_PAIRS),
    )


def majority_read(
    luminance: np.ndarray,
    positions: list[tuple[int, int]],
    repetition: int,
) -> tuple[list[int], list[float]]:
    """Collapse repeated block reads using majority voting."""
    bits: list[int] = []
    margins: list[float] = []

    for start in range(0, len(positions), repetition):
        group = positions[start:start + repetition]

        ones = sum(
            read_block(luminance, y, x)[0]
            for y, x in group
        )

        count = len(group)

        bits.append(1 if ones * 2 >= count else 0)
        margins.append(abs(2 * ones - count) / max(1, count))

    return bits, margins


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

def load_encoder_image(path: str | Path) -> Image.Image:
    """Load and normalize the carrier image."""
    image = Image.open(path).convert("RGB")

    if image.size != TARGET_SIZE:
        print(
            f"Resizing carrier: "
            f"{image.width}x{image.height} -> "
            f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
        )

        image = image.resize(
            TARGET_SIZE,
            Image.Resampling.LANCZOS,
        )

    return image


def get_luminance_channels(
    image: Image.Image,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an image to YCbCr and return its three channels."""
    ycbcr = np.asarray(
        image.convert("YCbCr"),
        dtype=np.float64,
    )

    return (
        ycbcr[:, :, 0],
        ycbcr[:, :, 1],
        ycbcr[:, :, 2],
    )


def save_ycbcr_image(
    path: str | Path,
    luminance: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
) -> None:
    """Reconstruct an RGB image while preserving Cb/Cr."""
    ycbcr = np.stack([luminance, cb, cr], axis=2)

    image = Image.fromarray(
        np.clip(np.rint(ycbcr), 0, 255).astype(np.uint8),
        "YCbCr",
    ).convert("RGB")

    image.save(path, format="PNG", optimize=False)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode(source: str, destination: str, message: str) -> None:
    """Embed message into an image."""
    image = load_encoder_image(source)
    width, height = image.size

    luminance, cb, cr = get_luminance_channels(image)

    header, payload = make_payload(message)

    # BCH + interleaving.
    header_bits = bytes_to_bits(header)
    header_coded = interleave(bch_encode(header_bits))

    payload_bits = bytes_to_bits(payload)
    payload_coded = interleave(bch_encode(payload_bits))

    # Repetition coding.
    header_stream = repeat_bits(
        header_coded,
        HEADER_REPETITION,
    )

    payload_stream = repeat_bits(
        payload_coded,
        PAYLOAD_REPETITION,
    )

    stream = header_stream + payload_stream

    blocks = block_order(width, height)

    if len(stream) > len(blocks):
        raise ValueError(
            f"Message too long: needs {len(stream)} blocks, "
            f"image has {len(blocks)}. "
            f"Approx max: {capacity(width, height)} bytes."
        )

    # Embed into luminance only.
    for (y, x), bit in zip(blocks, stream):
        embed_block(luminance, y, x, bit)

    save_ycbcr_image(
        destination,
        luminance,
        cb,
        cr,
    )

    print(f"Image: {width}x{height}")
    print(f"Message: {len(payload)} UTF-8 bytes")
    print(
        f"Header raw: {len(header_bits)} bits -> "
        f"BCH: {len(header_coded)} bits, "
        f"repetition={HEADER_REPETITION}"
    )
    print(
        f"Payload raw: {len(payload_bits)} bits -> "
        f"BCH: {len(payload_coded)} bits, "
        f"repetition={PAYLOAD_REPETITION}"
    )
    print(f"Blocks used: {len(stream)}")
    print(f"Blocks available: {len(blocks)}")
    print(f"Delta: {DELTA}")
    print(f"Approx max message: {capacity(width, height)} bytes")
    print(
        f"Block order: public deterministic shuffle "
        f"(seed 0x{SHUFFLE_SEED:08x})"
    )
    print("Color preserved: Y modified, Cb/Cr untouched")
    print("Header and payload both BCH-protected")
    print(f"Coefficient-pair votes per block: {len(COEFFICIENT_PAIRS)}")
    print(f"Saved: {destination}")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode(source: str) -> int:
    """Extract and validate a hidden message."""
    image = Image.open(source).convert("YCbCr")
    width, height = image.size

    if (width, height) != TARGET_SIZE:
        print(
            f"ERROR: Expected {TARGET_SIZE[0]}x{TARGET_SIZE[1]} "
            f"image, got {width}x{height}."
        )
        return 1

    ycbcr = np.asarray(image, dtype=np.float64)
    luminance = ycbcr[:, :, 0]

    blocks = block_order(width, height)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    header_raw_bits = HEADER_SIZE * 8
    header_coded_bits = coded_length(header_raw_bits)
    header_blocks = header_coded_bits * HEADER_REPETITION

    if len(blocks) < header_blocks:
        raise ValueError("Image is too small for the header.")

    header_read, _ = majority_read(
        luminance,
        blocks[:header_blocks],
        HEADER_REPETITION,
    )

    header_decoded = bch_decode(
        deinterleave(header_read),
        header_raw_bits,
    )

    header = bits_to_bytes(header_decoded.bits)

    print(f"Image: {width}x{height}")
    print(f"Header repetition: {HEADER_REPETITION}")
    print(f"Payload repetition: {PAYLOAD_REPETITION}")
    print(f"Delta: {DELTA}")
    print("Block order: public deterministic shuffle")
    print("Color image: Y channel decoded")
    print("Header and payload both BCH-protected")
    print(
        f"Coefficient-pair votes per block: "
        f"{len(COEFFICIENT_PAIRS)}"
    )
    print()

    print(f"Header bytes: {header.hex()}")
    print(f"Header BCH codewords: {header_decoded.codewords}")
    print(
        f"Header corrected codewords: "
        f"{header_decoded.corrected_codewords}"
    )
    print(
        f"Header uncorrectable codewords: "
        f"{header_decoded.uncorrectable_codewords}"
    )

    if (
        len(header) != HEADER_SIZE
        or header_decoded.uncorrectable_codewords
    ):
        print("HEADER CORRUPTED")
        return 1

    magic, message_length, expected_crc = struct.unpack(
        HEADER_FORMAT,
        header,
    )

    print(f"Decoded magic: {magic!r}")
    print(f"Reported message length: {message_length} bytes")
    print(f"Expected CRC: {expected_crc:08x}")

    if magic != MAGIC:
        print("MAGIC FAILED")
        print(f"Expected: {MAGIC!r}")
        return 1

    if message_length > MAX_MESSAGE_BYTES:
        print(
            f"MESSAGE LENGTH FAILED: "
            f"{message_length} > {MAX_MESSAGE_BYTES}"
        )
        return 1

    # ------------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------------

    payload_raw_bits = message_length * 8
    payload_coded_bits = coded_length(payload_raw_bits)
    payload_blocks = payload_coded_bits * PAYLOAD_REPETITION

    start = header_blocks
    end = start + payload_blocks

    if end > len(blocks):
        print("PAYLOAD TOO LARGE FOR THIS IMAGE")
        return 1

    payload_read, margins = majority_read(
        luminance,
        blocks[start:end],
        PAYLOAD_REPETITION,
    )

    payload_decoded = bch_decode(
        deinterleave(payload_read),
        payload_raw_bits,
    )

    data = bits_to_bytes(payload_decoded.bits)[:message_length]

    actual_crc = zlib.crc32(data) & 0xFFFFFFFF

    low_margin_count = sum(
        margin < 0.20
        for margin in margins
    )

    print(f"Payload BCH codewords: {payload_decoded.codewords}")
    print(
        f"Payload corrected codewords: "
        f"{payload_decoded.corrected_codewords}"
    )
    print(
        f"Payload uncorrectable codewords: "
        f"{payload_decoded.uncorrectable_codewords}"
    )
    print(
        f"Low-margin payload bits: "
        f"{low_margin_count}/{len(margins)}"
    )
    print(f"Actual CRC: {actual_crc:08x}")
    print()

    if (
        payload_decoded.uncorrectable_codewords
        or actual_crc != expected_crc
    ):
        print("PAYLOAD CORRUPTED")

        if payload_decoded.uncorrectable_codewords:
            print("BCH detected uncorrectable payload errors.")
        else:
            print("CRC failed. The payload still contains errors.")

        return 2

    try:
        message = data.decode("utf-8")
    except UnicodeDecodeError:
        print("PAYLOAD CORRUPTED")
        print("UTF-8 decoding failed.")
        return 2

    print("SUCCESS")
    print(f"Message: {message}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    encode_parser = subparsers.add_parser(
        "encode",
        help="Embed a message into an image.",
    )
    encode_parser.add_argument(
        "source",
        help="Source carrier image.",
    )
    encode_parser.add_argument(
        "destination",
        help="Output PNG image.",
    )
    encode_parser.add_argument(
        "message",
        help="UTF-8 message to hide.",
    )

    decode_parser = subparsers.add_parser(
        "decode",
        help="Extract a message from an image.",
    )
    decode_parser.add_argument(
        "source",
        help="Encoded image.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "encode":
            encode(
                args.source,
                args.destination,
                args.message,
            )
            return 0

        if args.command == "decode":
            return decode(args.source)

        parser.error(f"Unknown command: {args.command}")

    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())