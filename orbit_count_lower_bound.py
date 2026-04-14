#!/usr/bin/env python3
"""
Lower bound estimate for the number of restriction orbits (paper main.tex, Eq. after line 261):

    sum_{d=0}^{lm}  p^(l m d) / (|GL(l,F_p)| |GL(m,F_p)| |GL(d,F_p)| * |T|)

where |GL(n,F_p)| = prod_{k=0}^{n-1} (p^n - p^k), and |T| = 2 when the transpose
C_2 symmetry is counted (paper: l = m = n), else |T| = 1.

The `transpose` flag sets whether to divide by 2 for that symmetry factor.
"""

from __future__ import annotations

import argparse
from fractions import Fraction


def gl_order(n: int, p: int) -> int:
    """Order of GL(n, F_p) for prime p; GL(0) is trivial of order 1."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    if p < 2:
        raise ValueError("p must be >= 2")
    r = 1
    for k in range(n):
        r *= p**n - p**k
    return r


def orbit_count_lower_bound(l: int, m: int, p: int, transpose: bool) -> Fraction:
    """
    Rational value of the orbit-count lower bound sum.

    Parameters
    ----------
    l, m : positive integers (matrix factor dimensions)
    p : prime (field size)
    transpose : if True, include |T| = 2 in the denominator (transpose quotient);
                if False, use |T| = 1.
    """
    if l < 1 or m < 1:
        raise ValueError("l and m must be positive")
    denom_sym = 2 if transpose else 1
    gl_l = gl_order(l, p)
    gl_m = gl_order(m, p)
    total = Fraction(0, 1)
    for d in range(l * m + 1):
        num = p ** (l * m * d)
        den = gl_l * gl_m * gl_order(d, p) * denom_sym
        total += Fraction(num, den)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orbit-count lower bound from the matrix-multiplication search paper."
    )
    parser.add_argument("l", type=int, help="Rows of the A matrix")
    parser.add_argument("m", type=int, help="Columns of the A matrix")
    parser.add_argument("p", type=int, help="Field size (prime)")
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Include transpose symmetry |T|=2 in the denominator (paper: l=m=n)",
    )
    args = parser.parse_args()

    bound = orbit_count_lower_bound(args.l, args.m, args.p, args.transpose)
    print(bound)
    print(f"{float(bound):e}")


if __name__ == "__main__":
    main()
