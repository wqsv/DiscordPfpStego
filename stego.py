#!/usr/bin/env python3
"""Hide short UTF-8 messages in an image using DCT coefficient differences.

Examples:
    python stego.py encode carrier.png encoded.png "Hello from Discord."
    python stego.py decode downloaded.webp

"""

import argparse
import math
import struct
import zlib
from dataclasses import dataclass
import numpy as np
import cv2
from collections import Counter

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Settings. These values must match between encoder and decoder.
# ---------------------------------------------------------------------------

BLOCK_SIZE = 8
IMAGE_SIZE = (1024, 1024)
DELTA = 35.0
HEADER_REPETITION = 15
PAYLOAD_REPETITION = 20
SHUFFLE_SEED = 0x6D2B79F5

MAGIC = b"BCH7"
HEADER_FORMAT = ">4sHI"  # magic, payload length, CRC32
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_MESSAGE_BYTES = 60

# BCH(31, 21) t=2
BCH_N = 31
BCH_K = 21
BCH_R = 10
BCH_GENERATOR = 0x769

# One coefficient pair per 8x8 block: (2, 3) compared with (3, 2).
COEFFICIENT_A = (2, 3)
COEFFICIENT_B = (3, 2)

class HeaderError(ValueError):
    """Failure while reading or validating the fixed-size header."""
    exit_code = 1


class PayloadError(ValueError):
    """Failure while reading or validating the variable-length payload."""
    exit_code = 2

# ---------------------------------------------------------------------------
# DCT helpers
# ---------------------------------------------------------------------------

def make_dct_matrix():
    """Build the orthonormal 8x8 DCT matrix once."""
    matrix = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)

    for u in range(BLOCK_SIZE):
        scale = math.sqrt(1 / BLOCK_SIZE) if u == 0 else math.sqrt(2 / BLOCK_SIZE)
        for x in range(BLOCK_SIZE):
            matrix[u, x] = scale * math.cos(
                math.pi * (2 * x + 1) * u / (2 * BLOCK_SIZE)
            )

    return matrix


DCT_MATRIX = make_dct_matrix()


def dct(block):
    return DCT_MATRIX @ block @ DCT_MATRIX.T


def inverse_dct(coefficients):
    return DCT_MATRIX.T @ coefficients @ DCT_MATRIX


@dataclass
class ClippingStats:
    total_blocks: int = 0
    clipped_blocks: int = 0
    clipped_pixels: int = 0
    below_zero: int = 0
    above_255: int = 0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    overshoot_total: float = 0.0
    clipped_values: Counter = None

    def __post_init__(self):
        if self.clipped_values is None:
            self.clipped_values = Counter()

    def record(self, rounded, clipping_mask):
        self.total_blocks += 1

        self.min_value = min(self.min_value, float(rounded.min()))
        self.max_value = max(self.max_value, float(rounded.max()))

        if not np.any(clipping_mask):
            return

        self.clipped_blocks += 1
        self.clipped_pixels += int(np.count_nonzero(clipping_mask))

        self.below_zero += int(np.count_nonzero(rounded < 0))
        self.above_255 += int(np.count_nonzero(rounded > 255))

        self.overshoot_total += float(
            np.maximum(-rounded, 0).sum()
            + np.maximum(rounded - 255, 0).sum()
        )

        # Track how often each out-of-range integer occurs.
        values = rounded[clipping_mask].astype(np.int64)
        self.clipped_values.update(values.tolist())

    def report(self):
        print("\n--- Clipping Statistics ---")
        print(f"Blocks processed:       {self.total_blocks:,}")
        print(f"Blocks with clipping:   {self.clipped_blocks:,}")
        print(f"Total clipped pixels:   {self.clipped_pixels:,}")
        print(f"Pixels below 0:         {self.below_zero:,}")
        print(f"Pixels above 255:       {self.above_255:,}")
        print(f"Minimum pre-clip value:  {self.min_value:.0f}")
        print(f"Maximum pre-clip value:  {self.max_value:.0f}")
        print(f"Total clipping overshoot: {self.overshoot_total:.0f}")

        if self.clipped_values:
            print("\nMost frequent clipped values:")
            for value, count in self.clipped_values.most_common(10):
                print(f"  {value:>5}: {count:,} occurrences")


def embed_bit_in_block(luminance, y, x, bit, stats=None):
    """Embed a bit while preserving existing DCT coefficients when possible."""
    block = luminance[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
    coefficients = dct(block)

    a = float(coefficients[COEFFICIENT_A])
    b = float(coefficients[COEFFICIENT_B])
    difference = a - b

    target = DELTA if bit else -DELTA

    # Only modify the coefficients if the current difference doesn't already represent the desired bit with sufficient margin.
    needs_adjustment = (
        difference < target if bit else difference > target
    )

    if needs_adjustment:
        adjustment = (target - difference) / 2.0
        coefficients[COEFFICIENT_A] = a + adjustment
        coefficients[COEFFICIENT_B] = b - adjustment

    pixels = inverse_dct(coefficients)
    rounded = np.rint(pixels)

    clipping_mask = (rounded < 0) | (rounded > 255)

    if stats is not None:
        stats.record(rounded, clipping_mask)

    encoded = np.clip(rounded, 0, 255)
    luminance[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = encoded

def read_bit_from_block(luminance, y, x):
    """Read a bit and return its coefficient-difference magnitude as confidence."""
    block = luminance[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
    coefficients = dct(block)
    difference = float(coefficients[COEFFICIENT_A] - coefficients[COEFFICIENT_B])
    return (1 if difference >= 0 else 0), abs(difference)


# ---------------------------------------------------------------------------
# Bytes, bits, BCH, and interleaving
# ---------------------------------------------------------------------------

def bytes_to_bits(data):
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bits_to_bytes(bits):
    result = bytearray()
    for start in range(0, len(bits) - 7, 8):
        value = 0
        for bit in bits[start:start + 8]:
            value = (value << 1) | int(bit)
        result.append(value)
    return bytes(result)


def polynomial_mod(value):
    """Reduce a polynomial bit-field modulo the BCH generator polynomial."""
    while value and value.bit_length() - 1 >= BCH_R:
        shift = value.bit_length() - 1 - BCH_R
        value ^= BCH_GENERATOR << shift
    return value


def encode_bch_word(data):
    data &= (1 << BCH_K) - 1
    shifted = data << BCH_R
    return shifted | polynomial_mod(shifted)


def make_error_lookup():
    """Map syndromes to error patterns containing zero, one, or two flipped bits."""
    lookup = {0: 0}

    for first in range(BCH_N):
        error = 1 << first
        syndrome = polynomial_mod(error)
        if syndrome in lookup and lookup[syndrome] != error:
            raise RuntimeError("BCH generator produced a syndrome collision")
        lookup[syndrome] = error

    for first in range(BCH_N):
        for second in range(first + 1, BCH_N):
            error = (1 << first) | (1 << second)
            syndrome = polynomial_mod(error)
            if syndrome in lookup and lookup[syndrome] != error:
                raise RuntimeError("BCH generator produced a syndrome collision")
            lookup[syndrome] = error

    return lookup


ERROR_BY_SYNDROME = make_error_lookup()


def bch_encode(bits):
    """Encode a bit stream in 21-bit chunks, padding the last chunk with zeros."""
    encoded = []
    for start in range(0, len(bits), BCH_K):
        chunk = list(bits[start:start + BCH_K])
        chunk.extend([0] * (BCH_K - len(chunk)))

        value = 0
        for bit in chunk:
            value = (value << 1) | int(bit)

        word = encode_bch_word(value)
        encoded.extend((word >> (BCH_N - 1 - i)) & 1 for i in range(BCH_N))

    return encoded


@dataclass
class BCHResult:
    bits: list
    corrected_codewords: int
    uncorrectable_codewords: int
    codewords: int


def bch_decode(bits, expected_bit_count):
    """Decode BCH words. If any word is uncorrectable, callers must reject the result."""
    output = []
    corrected = 0
    uncorrectable = 0
    codeword_count = 0

    for start in range(0, len(bits), BCH_N):
        chunk = bits[start:start + BCH_N]
        if len(chunk) != BCH_N:
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

        if polynomial_mod(word) != 0:
            uncorrectable += 1
            continue

        data = word >> BCH_R
        output.extend((data >> (BCH_K - 1 - i)) & 1 for i in range(BCH_K))

    return BCHResult(output[:expected_bit_count], corrected, uncorrectable, codeword_count)


def coded_length(raw_bit_count):
    words = (raw_bit_count + BCH_K - 1) // BCH_K
    return words * BCH_N


def permutation(size):
    """Generate the same deterministic permutation used by the encoder/decoder."""
    order = list(range(size))
    state = SHUFFLE_SEED & 0xFFFFFFFF

    for index in range(size - 1, 0, -1):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        other = state % (index + 1)
        order[index], order[other] = order[other], order[index]

    return order


def interleave(bits):
    order = permutation(len(bits))
    output = [0] * len(bits)
    for index, bit in enumerate(bits):
        output[order[index]] = int(bit)
    return output


def deinterleave(bits):
    order = permutation(len(bits))
    return [int(bits[order[index]]) for index in range(len(bits))]


def repeat_bits(bits, count):
    return [int(bit) for bit in bits for _ in range(count)]


def majority_read(luminance, blocks, repetition):
    """Vote over repeated observations. A tie resolves to 0, by explicit choice."""
    recovered = []
    margins = []

    for start in range(0, len(blocks), repetition):
        group = blocks[start:start + repetition]
        votes = [read_bit_from_block(luminance, y, x)[0] for y, x in group]
        ones = sum(votes)
        count = len(votes)
        recovered.append(1 if ones * 2 > count else 0)
        margins.append(abs(2 * ones - count) / max(1, count))

    return recovered, margins


# ---------------------------------------------------------------------------
# Image handling and message format
# ---------------------------------------------------------------------------

def load_carrier(path):
    with Image.open(path) as source:
        has_transparency = ("transparency" in source.info or "A" in source.getbands())
        image = source.convert("RGBA")

    if has_transparency:
        print("\nNote: this image appears to have transparency.\nPress Enter for a black background or type 'w' for a white background.")
        choice = input("> ").strip().lower()

        color = (255, 255, 255, 255) if choice == "w" else (0, 0, 0, 255)
        background = Image.new("RGBA", image.size, color)
        image = Image.alpha_composite(background, image)

    image = image.convert("RGB")
    if image.size != IMAGE_SIZE:
        print(f"Resizing carrier: {image.width}x{image.height} -> {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return image


def split_ycbcr(image):
    channels = np.asarray(image.convert("YCbCr"), dtype=np.float64)
    return channels[:, :, 0], channels[:, :, 1], channels[:, :, 2]


def save_ycbcr(path, luminance, cb, cr):
    channels = np.stack([luminance, cb, cr], axis=2)
    channels = np.clip(np.rint(channels), 0, 255).astype(np.uint8)
    Image.fromarray(channels, "YCbCr").convert("RGB").save(path, format="PNG")


def make_header_and_payload(message):
    payload = message.encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Message is {len(payload)} UTF-8 bytes; maximum is {MAX_MESSAGE_BYTES}."
        )

    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    header = struct.pack(HEADER_FORMAT, MAGIC, len(payload), checksum)
    return header, payload


def block_order(width, height):
    blocks = [
        (y, x)
        for y in range(0, height, BLOCK_SIZE)
        for x in range(0, width, BLOCK_SIZE)
    ]

    state = SHUFFLE_SEED & 0xFFFFFFFF
    for index in range(len(blocks) - 1, 0, -1):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        other = state % (index + 1)
        blocks[index], blocks[other] = blocks[other], blocks[index]

    return blocks


def estimate_capacity(width, height):
    block_count = (width // BLOCK_SIZE) * (height // BLOCK_SIZE)
    header_coded = coded_length(HEADER_SIZE * 8)
    header_blocks = header_coded * HEADER_REPETITION
    remaining = block_count - header_blocks
    if remaining <= 0:
        return 0

    payload_codewords = (remaining // PAYLOAD_REPETITION) // BCH_N
    return min(MAX_MESSAGE_BYTES, (payload_codewords * BCH_K) // 8)


def prepare_stream(data, repetition):
    return repeat_bits(interleave(bch_encode(bytes_to_bits(data))), repetition)


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def encode(source, destination, message):
    stats = ClippingStats()


    image = load_carrier(source)
    width, height = image.size
    luminance, cb, cr = split_ycbcr(image)

    header, payload = make_header_and_payload(message)
    header_stream = prepare_stream(header, HEADER_REPETITION)
    payload_stream = prepare_stream(payload, PAYLOAD_REPETITION)
    stream = header_stream + payload_stream

    blocks = block_order(width, height)
    if len(stream) > len(blocks):
        raise ValueError(f"Message needs {len(stream)} blocks; image has {len(blocks)}.")

    for (y, x), bit in zip(blocks, stream):
        embed_bit_in_block(luminance, y, x, bit, stats)
    stats.report()

    save_ycbcr(destination, luminance, cb, cr)

    print(f"Message: {len(payload)} UTF-8 bytes")
    print(f"Blocks used: {len(stream)}")
    print(f"Blocks available: {len(blocks)}")
    print(f"Approximate maximum payload: {estimate_capacity(width, height)} bytes")
    print(f"Saved: {destination}")


def decode(source):
    image = Image.open(source).convert("YCbCr")
    width, height = image.size
    if (width, height) != IMAGE_SIZE:
        raise HeaderError(f"Expected a {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} image, got {width}x{height}.")

    channels = np.asarray(image, dtype=np.float64)
    luminance = channels[:, :, 0]
    blocks = block_order(width, height)

    # Recover the fixed-size header first so we know how long the payload is.
    header_bit_count = HEADER_SIZE * 8
    header_coded_count = coded_length(header_bit_count)
    header_block_count = header_coded_count * HEADER_REPETITION

    header_votes, _ = majority_read(
        luminance, blocks[:header_block_count], HEADER_REPETITION
    )
    header_result = bch_decode(deinterleave(header_votes), header_bit_count)
    header = bits_to_bytes(header_result.bits)

    print(f"Header corrected codewords: {header_result.corrected_codewords}")
    print(f"Header uncorrectable codewords: {header_result.uncorrectable_codewords}")

    if len(header) != HEADER_SIZE or header_result.uncorrectable_codewords:
        raise HeaderError("Header corrupted; could not decode all header codewords.")

    magic, message_length, expected_crc = struct.unpack(HEADER_FORMAT, header)

    if magic != MAGIC:
        raise HeaderError("Header magic value does not match.")
    if message_length > MAX_MESSAGE_BYTES:
        raise HeaderError(f"Payload length exceeds {MAX_MESSAGE_BYTES} bytes.")

    # Now recover just the number of repeated bits required for this payload.
    payload_bit_count = message_length * 8
    payload_coded_count = coded_length(payload_bit_count)
    payload_block_count = payload_coded_count * PAYLOAD_REPETITION
    start = header_block_count
    end = start + payload_block_count
    if end > len(blocks):
        raise HeaderError("Payload length exceeds the available image capacity.")

    payload_votes, margins = majority_read(
        luminance, blocks[start:end], PAYLOAD_REPETITION
    )
    payload_result = bch_decode(deinterleave(payload_votes), payload_bit_count)
    payload = bits_to_bytes(payload_result.bits)[:message_length]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF

    print(f"Payload corrected codewords: {payload_result.corrected_codewords}")
    print(f"Payload uncorrectable codewords: {payload_result.uncorrectable_codewords}")
    print(f"Low-margin payload bits: {sum(m < 0.20 for m in margins)}/{len(margins)}")

    if payload_result.uncorrectable_codewords:
        raise PayloadError(
            f"Payload corrupted: BCH could not correct "
            f"{payload_result.uncorrectable_codewords} codeword(s)."
        )

    if actual_crc != expected_crc:
        raise PayloadError(
            f"Payload corrupted: CRC32 mismatch "
            f"(expected {expected_crc:08x}, got {actual_crc:08x})."
        )

    try:
        message = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PayloadError("Recovered payload is not valid UTF-8.") from exc

    print("SUCCESS")
    print(f"Message: {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    encode_parser = commands.add_parser("encode", help="Hide a message in an image.")
    encode_parser.add_argument("source", help="Input carrier image")
    encode_parser.add_argument("destination", help="Output PNG image")
    encode_parser.add_argument("message", help="Message to hide")

    decode_parser = commands.add_parser("decode", help="Recover a message from an image.")
    decode_parser.add_argument("source", help="Processed image containing the message")
    args = parser.parse_args()

    try:
        if args.command == "encode":
            encode(args.source, args.destination, args.message)
        else:
            decode(args.source)
    except (HeaderError, PayloadError) as exc:
        parser.exit(exc.exit_code, f"ERROR: {exc}\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
