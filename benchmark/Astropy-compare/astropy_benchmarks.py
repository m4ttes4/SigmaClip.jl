#!/usr/bin/env python3

import csv
import statistics
import timeit
import warnings
from pathlib import Path

import numpy as np
from astropy.stats import sigma_clip

warnings.filterwarnings("ignore", message="Input data contains invalid values.*")

OUTPUT = Path(__file__).with_name("astropy_results.csv")
SIGMA = 3.0
MAXITERS = 5
SEED = 51461393
VECTOR_SIZES = (128, 1024, 10000, 100000)
MATRIX_SIZES = ((32, 32), (128, 128), (512, 512))


def make_data(shape, seed):
    data = np.random.default_rng(seed).standard_normal(shape)
    values = data.reshape(-1)
    values[0::101] += 8.0
    values[50::173] -= 8.0
    values[len(values) // 3] = np.nan
    values[2 * len(values) // 3] = np.inf
    return data


def measure(data, number):
    call = lambda: sigma_clip(
        data,
        sigma_lower=SIGMA,
        sigma_upper=SIGMA,
        maxiters=MAXITERS,
        cenfunc="median",
        stdfunc="mad_std",
        masked=True,
        copy=True,
    )
    call()
    times = timeit.repeat(call, repeat=5, number=number)
    return statistics.median(times) * 1e9 / number, min(times) * 1e9 / number


with OUTPUT.open("w", newline="") as file:
    output = csv.writer(file)
    output.writerow(
        (
            "algorithm",
            "family",
            "shape",
            "elements",
            "mode",
            "median_ns",
            "min_ns",
            "samples",
            "seed",
            "sigma_lower",
            "sigma_upper",
            "maxiters",
            "center",
            "spread",
            "memory_bytes",
            "allocations",
        )
    )

    for n in VECTOR_SIZES:
        data = make_data((n,), SEED + n)
        number = 50 if n <= 10000 else 15
        median_ns, minimum_ns = measure(data, number)
        output.writerow(
            (
                "astropy.stats.sigma_clip",
                "vector",
                n,
                n,
                "out_of_place",
                median_ns,
                minimum_ns,
                5,
                SEED + n,
                SIGMA,
                SIGMA,
                MAXITERS,
                "median",
                "mad_std",
                "",
                "",
            )
        )

    for shape in MATRIX_SIZES:
        data = make_data(shape, SEED + int(np.prod(shape)))
        number = 50 if data.size <= 10000 else 15
        median_ns, minimum_ns = measure(data, number)
        output.writerow(
            (
                "astropy.stats.sigma_clip",
                "matrix",
                "x".join(map(str, shape)),
                data.size,
                "out_of_place",
                median_ns,
                minimum_ns,
                5,
                SEED + int(np.prod(shape)),
                SIGMA,
                SIGMA,
                MAXITERS,
                "median",
                "mad_std",
                "",
                "",
            )
        )

print(f"Wrote {OUTPUT}")
