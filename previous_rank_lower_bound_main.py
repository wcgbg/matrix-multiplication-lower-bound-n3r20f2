import argparse


def hk71(n0, n1, n2):
    if n0 == 2 and n1 == 2 and n2 >= 2:
        return ((7 * n2 + 1) // 2, -1971, "hk71")
    if n0 == 3 and n1 == 2 and n2 == 3:
        return (15, -1971, "hk71")
    if n0 == 3 and n1 == 2 and n2 == 4:
        return (19 - 1e-3, -1971, "hk71")  # no proof of 19
    return (0, 0, "")


def blaser1999lower(l, m, n):
    if n >= l and l >= 2:
        return (l * m + m * n + l - m + n - 3, -1999, "blaser1999lower")
    return (0, 0, "")


def blaser1999fivehalf(n0, n1, n2):
    if n0 == n1 == n2 and n0 >= 3:
        return ((5 * n0**2 + 1) // 2 - 3 * n0, -1999, "blaser1999fivehalf")
    return (0, 0, "")


def blaser2003complexity(n, m, n2):
    if n == n2 and m >= n and n >= 3:
        return (2 * m * n + 2 * n - m - 2, -2003, "blaser2003complexity")
    return (0, 0, "")


def rank_lower_bound_n012(n0, n1, n2):
    return max(
        hk71(n0, n1, n2),
        blaser1999lower(n0, n1, n2),
        blaser1999fivehalf(n0, n1, n2),
        blaser2003complexity(n0, n1, n2),
    )


def rank_lower_bound(n0, n1, n2):
    return max(
        rank_lower_bound_n012(n0, n1, n2),
        rank_lower_bound_n012(n0, n2, n1),
        rank_lower_bound_n012(n1, n0, n2),
        rank_lower_bound_n012(n1, n2, n0),
        rank_lower_bound_n012(n2, n0, n1),
        rank_lower_bound_n012(n2, n1, n0),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Previous rank lower bounds for matrix multiplication over field F_2."
    )
    parser.add_argument("n0", type=int)
    parser.add_argument("n1", type=int)
    parser.add_argument("n2", type=int)
    args = parser.parse_args()
    n0, n1, n2 = args.n0, args.n1, args.n2
    assert n0 > 0
    assert n1 > 0
    assert n2 > 0

    print(*rank_lower_bound(n0, n1, n2))


if __name__ == "__main__":
    main()
