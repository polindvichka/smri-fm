"""Download a targeted subset of FOMO_with_dwi webdataset shards from
Cloudflare R2 into `datasets/FOMO_with_dwi/` (gitignored, see root
`.gitignore`'s `/datasets/FOMO_with_dwi/` entry).

This is **not** a unit test -- run it directly with any Python that has
`boto3`/`python-dotenv` installed (the tt-metal `python_env` this project
already uses for `smri_mae_tt` has both). It is a small, reusable CLI so
future "grab N more shards" requests don't require re-deriving the
bucket/prefix/credential-loading pattern from scratch.

## Source

Bucket `medarc` (R2_BUCKET in `tenstorrent/.env`), key prefix
`mihir-stuff/processed/FOMO300/wds_with_dwi/`, shard names
`shard.NNNNNN.tar` (zero-padded 6 digits), sparse webdataset format,
150 samples/shard (confirmed by `tarfile` listing -- 450 tar members in
`shard.000000.tar` / 3 members per sample), ~1800 shards total in the
bucket (~270k samples).

**Do not download anywhere near the whole bucket.** This project only ever
needs a small local subset for TT-NN development/testing -- always pass an
explicit, small `--keys`/`--count` value.

## Bandwidth

Downloads stream via a chunked `get_object`/`iter_chunks` read (never
boto3's default multipart `download_file`, which can open several
concurrent range-request connections per file) and are always strictly
sequential -- one shard at a time, never multiple concurrent connections.
By default throughput is uncapped (`--max-mbps` unset / `None`); pass
`--max-mbps <N>` to cap total download throughput to N MB/s (a shared
rate limiter across the whole run) if a run is saturating the host's link
-- e.g. `--max-mbps 5` for ~40 Mbps. Do not add parallel downloads.

## Credentials

Loaded via `python-dotenv` from `tenstorrent/.env` (gitignored):
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT_URL`, `R2_BUCKET`.
Never hardcoded here, never printed, never committed.

## Usage

List the next available shard keys after what's already local, without
downloading anything (targeted single `list_objects_v2` call, using
`StartAfter` so it only reads forward from the given shard rather than
walking the whole ~1800-shard prefix):

    python download_r2_shards.py --list --after shard.000003.tar --count 80

Download a specific, explicit set of shards:

    python download_r2_shards.py --keys shard.000004.tar shard.000005.tar \\
        --dest datasets/FOMO_with_dwi

Download the next N shards after a given shard (list + download in one go):

    python download_r2_shards.py --after shard.000003.tar --count 63 \\
        --dest datasets/FOMO_with_dwi
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ENV_FILE = _REPO_ROOT / "tenstorrent" / ".env"
DEFAULT_DEST_DIR = _REPO_ROOT / "datasets" / "FOMO_with_dwi"
PREFIX = "mihir-stuff/processed/FOMO300/wds_with_dwi/"


def _load_credentials(env_file: Path) -> dict[str, str]:
    from dotenv import load_dotenv

    if not env_file.is_file():
        raise FileNotFoundError(
            f"credentials file not found: {env_file} -- expected R2_ACCESS_KEY_ID/"
            "R2_SECRET_ACCESS_KEY/R2_ENDPOINT_URL/R2_BUCKET in a .env file there"
        )
    load_dotenv(env_file)
    required = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL", "R2_BUCKET")
    creds = {k: os.environ.get(k) for k in required}
    missing = [k for k, v in creds.items() if not v]
    if missing:
        raise RuntimeError(f"missing required env var(s) in {env_file}: {', '.join(missing)}")
    return creds


def _make_client(creds: dict[str, str]):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=creds["R2_ENDPOINT_URL"],
        aws_access_key_id=creds["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=creds["R2_SECRET_ACCESS_KEY"],
    )


def list_next_shard_keys(client, bucket: str, *, after: str | None, count: int) -> list[str]:
    """Targeted listing: a single `list_objects_v2` call (or a small,
    bounded number of paginated calls if `count` exceeds one page),
    scoped to `PREFIX` and starting right after `after` (a bare shard
    filename like "shard.000003.tar", or None to list from the start of
    the prefix). This intentionally never walks the whole ~1800-shard
    prefix -- it stops as soon as `count` keys have been collected.
    """
    start_after = PREFIX + after if after else None
    keys: list[str] = []
    continuation_token = None
    while len(keys) < count:
        kwargs = {"Bucket": bucket, "Prefix": PREFIX, "MaxKeys": min(1000, count - len(keys))}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        elif start_after:
            kwargs["StartAfter"] = start_after
        resp = client.list_objects_v2(**kwargs)
        keys.extend(obj["Key"] for obj in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        continuation_token = resp.get("NextContinuationToken")
    return keys[:count]


DEFAULT_MAX_MBPS = None  # unlimited by default -- pass --max-mbps (e.g. 5.0 for
# ~40 Mbps) to cap throughput if a run is saturating the host's link.
_CHUNK_BYTES = 512 * 1024  # 512 KiB read/write granularity (also the rate limiter's
# granularity when a --max-mbps cap is active)


class _RateLimiter:
    """Token-bucket-ish limiter: sleeps just enough after each chunk so the
    *cumulative* bytes/elapsed average never exceeds `max_bytes_per_sec`,
    across the whole run (shared across all shards in one `download_shards`
    call) -- not reset per-shard, so a fast shard can't "make up" bandwidth
    for a slow one.
    """

    def __init__(self, max_bytes_per_sec: float) -> None:
        self._max_bps = max_bytes_per_sec
        self._start = time.monotonic()
        self._sent = 0

    def throttle(self, n: int) -> None:
        self._sent += n
        elapsed = time.monotonic() - self._start
        expected = self._sent / self._max_bps
        if expected > elapsed:
            time.sleep(expected - elapsed)


def download_shards(
    client,
    bucket: str,
    keys: list[str],
    dest_dir: Path,
    *,
    max_mbps: float | None = DEFAULT_MAX_MBPS,
) -> tuple[int, float]:
    """Download each key to `dest_dir/<basename>`, skipping files already
    present (so re-running is idempotent). Returns (total_bytes, elapsed_s)
    for newly downloaded files only.

    Strictly sequential (one shard at a time, plain `for` loop -- never
    threaded/parallel), so this never opens multiple concurrent
    connections. `max_mbps=None` (the default) means uncapped throughput;
    pass a number to rate-limit to that many MB/s (a single `_RateLimiter`
    shared across every key in this call, so it never bursts past the cap
    even briefly between shards).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    start = time.monotonic()
    limiter = _RateLimiter(max_mbps * 1e6) if max_mbps is not None else None
    for key in keys:
        name = key.rsplit("/", 1)[-1]
        dest = dest_dir / name
        if dest.is_file():
            print(f"skip (already present): {name}")
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        t0 = time.monotonic()
        size = 0
        obj = client.get_object(Bucket=bucket, Key=key)
        with open(tmp, "wb") as f:
            for chunk in obj["Body"].iter_chunks(chunk_size=_CHUNK_BYTES):
                f.write(chunk)
                size += len(chunk)
                if limiter is not None:
                    limiter.throttle(len(chunk))
        tmp.rename(dest)
        total_bytes += size
        dt = time.monotonic() - t0
        print(f"downloaded {name}: {size / 1e6:.1f} MB in {dt:.1f}s ({size / 1e6 / max(dt, 1e-9):.1f} MB/s)")
    elapsed = time.monotonic() - start
    return total_bytes, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Explicit shard filenames to download (e.g. shard.000004.tar shard.000005.tar). "
        "Mutually exclusive with --after/--count listing mode.",
    )
    parser.add_argument(
        "--after",
        default=None,
        help="Bare shard filename (e.g. shard.000003.tar) to list/download starting after. "
        "Used with --count when --keys is not given.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of shard keys to list/download starting after --after.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list the resolved shard keys (via a targeted list_objects_v2 call); do not download.",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST_DIR),
        help=f"Destination directory (default: {DEFAULT_DEST_DIR}).",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help=f"Path to the .env file with R2 credentials (default: {DEFAULT_ENV_FILE}).",
    )
    parser.add_argument(
        "--max-mbps",
        type=float,
        default=DEFAULT_MAX_MBPS,
        help="Cap total download throughput to this many MB/s, averaged over the whole run "
        "and shared across all shards (default: unlimited -- pass e.g. 5.0 for ~40 Mbps if a "
        "run is saturating the host's link). Downloads are always strictly sequential (one "
        "shard at a time, never parallel connections) regardless of this value.",
    )
    args = parser.parse_args()

    if args.keys and (args.after or args.count):
        raise SystemExit("--keys is mutually exclusive with --after/--count")
    if not args.keys and args.count is None:
        raise SystemExit("must supply either --keys or --count (optionally with --after)")

    creds = _load_credentials(Path(args.env_file))
    client = _make_client(creds)
    bucket = creds["R2_BUCKET"]

    if args.keys:
        keys = [PREFIX + k for k in args.keys]
    else:
        keys = list_next_shard_keys(client, bucket, after=args.after, count=args.count)

    print(f"resolved {len(keys)} shard key(s):")
    for k in keys:
        print(f"  {k}")

    if args.list:
        return

    total_bytes, elapsed = download_shards(client, bucket, keys, Path(args.dest), max_mbps=args.max_mbps)
    print(f"\ndone: {total_bytes / 1e6:.1f} MB downloaded in {elapsed:.1f}s "
          f"({total_bytes / 1e6 / max(elapsed, 1e-9):.1f} MB/s average)")


if __name__ == "__main__":
    main()
