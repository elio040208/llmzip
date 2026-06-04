from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable


STATE_BITS = 32
FULL_RANGE = 1 << STATE_BITS
HALF_RANGE = FULL_RANGE >> 1
QUARTER_RANGE = HALF_RANGE >> 1
THREE_QUARTER_RANGE = QUARTER_RANGE * 3
MAX_RANGE = FULL_RANGE - 1


class BitWriter:
    def __init__(self) -> None:
        self._bytes = bytearray()
        self._current = 0
        self._count = 0

    def write(self, bit: int) -> None:
        self._current = (self._current << 1) | int(bit)
        self._count += 1
        if self._count == 8:
            self._bytes.append(self._current)
            self._current = 0
            self._count = 0

    def finish(self) -> bytes:
        if self._count:
            self._current <<= 8 - self._count
            self._bytes.append(self._current)
            self._current = 0
            self._count = 0
        return bytes(self._bytes)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 0

    def read(self) -> int:
        if self._byte_pos >= len(self._data):
            return 0
        bit = (self._data[self._byte_pos] >> (7 - self._bit_pos)) & 1
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._bit_pos = 0
            self._byte_pos += 1
        return bit


@dataclass
class ArithmeticEncoder:
    writer: BitWriter

    def __post_init__(self) -> None:
        self.low = 0
        self.high = MAX_RANGE
        self.pending_bits = 0

    def _write_bit_plus_pending(self, bit: int) -> None:
        self.writer.write(bit)
        inverse = 1 - bit
        while self.pending_bits:
            self.writer.write(inverse)
            self.pending_bits -= 1

    def encode(self, symbol: int, cumulative: list[int]) -> None:
        total = cumulative[-1]
        sym_low = cumulative[symbol]
        sym_high = cumulative[symbol + 1]
        width = self.high - self.low + 1
        self.high = self.low + (width * sym_high // total) - 1
        self.low = self.low + (width * sym_low // total)

        while True:
            if self.high < HALF_RANGE:
                self._write_bit_plus_pending(0)
            elif self.low >= HALF_RANGE:
                self._write_bit_plus_pending(1)
                self.low -= HALF_RANGE
                self.high -= HALF_RANGE
            elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
                self.pending_bits += 1
                self.low -= QUARTER_RANGE
                self.high -= QUARTER_RANGE
            else:
                break
            self.low = (self.low << 1) & MAX_RANGE
            self.high = ((self.high << 1) & MAX_RANGE) | 1

    def finish(self) -> bytes:
        self.pending_bits += 1
        if self.low < QUARTER_RANGE:
            self._write_bit_plus_pending(0)
        else:
            self._write_bit_plus_pending(1)
        return self.writer.finish()


@dataclass
class ArithmeticDecoder:
    reader: BitReader

    def __post_init__(self) -> None:
        self.low = 0
        self.high = MAX_RANGE
        self.value = 0
        for _ in range(STATE_BITS):
            self.value = (self.value << 1) | self.reader.read()

    def decode(self, cumulative: list[int]) -> int:
        total = cumulative[-1]
        width = self.high - self.low + 1
        scaled = (((self.value - self.low + 1) * total) - 1) // width
        symbol = bisect_right(cumulative, scaled) - 1
        if symbol < 0 or symbol + 1 >= len(cumulative):
            raise ValueError("Arithmetic decoder state is outside the cumulative table")

        sym_low = cumulative[symbol]
        sym_high = cumulative[symbol + 1]
        self.high = self.low + (width * sym_high // total) - 1
        self.low = self.low + (width * sym_low // total)

        while True:
            if self.high < HALF_RANGE:
                pass
            elif self.low >= HALF_RANGE:
                self.low -= HALF_RANGE
                self.high -= HALF_RANGE
                self.value -= HALF_RANGE
            elif self.low >= QUARTER_RANGE and self.high < THREE_QUARTER_RANGE:
                self.low -= QUARTER_RANGE
                self.high -= QUARTER_RANGE
                self.value -= QUARTER_RANGE
            else:
                break
            self.low = (self.low << 1) & MAX_RANGE
            self.high = ((self.high << 1) & MAX_RANGE) | 1
            self.value = ((self.value << 1) & MAX_RANGE) | self.reader.read()

        return symbol


def frequencies_to_cumulative(freqs: Iterable[int]) -> list[int]:
    total = 0
    cumulative = [0]
    for freq in freqs:
        total += int(freq)
        cumulative.append(total)
    if total <= 0:
        raise ValueError("Frequency table must have positive total")
    return cumulative
