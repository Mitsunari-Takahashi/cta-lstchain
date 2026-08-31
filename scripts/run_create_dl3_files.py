#!/usr/bin/env python

"""
Batch-run ``lstchain_create_dl3_file`` over real LST-1 DL2 data files for one
or more NSB levels, using the IRF FITS files produced for each NSB level by
``run_create_irf_files.py``.

This script is based on the workflow developed in the notebook
``DL3-analysis/notebooks/20260828_dl2_to_dl3_CrabNebula.ipynb`` (LabNotebook_CTA
repo), generalized to discover its input DL2 files and IRF directories instead
of relying on variables stashed by earlier notebook cells (``%store -r``).

Real DL2 data files can be provided in one of two ways:

- ``--input-dl2-base-dir`` (+ ``--input-dl2-glob``): the input files are
  discovered by globbing under a base directory. Real DL2 data files are
  assumed to be laid out so that the NSB tuning level applied to each run
  appears as an "nsb_tuning_<NSB>" path component, e.g.:

      <input-dl2-base-dir>/.../nsb_tuning_<NSB>/dl2_LST-1.Run#####.h5

  (adjust ``--input-dl2-glob`` if your directory layout differs; ``{nsb}``
  in the pattern is replaced with each ``--nsb`` value, formatted as "%.2f").

- ``--input-dl2``: an explicit, already-selected list of DL2 files (e.g. a
  quality-filtered good-run file list built elsewhere, such as with
  ``%store -r`` in a notebook). Each ``--nsb`` value's files are picked out
  of this list by matching ``--input-dl2-nsb-tag`` (formatted with that NSB
  value) against each file's path.

IRF FITS files are assumed to have been produced for each NSB level and laid
out (without any declination-node subdirectory) as:

    <irf-base-dir>/fits/<mc-tag>_nsb_<NSB>/irfs_theta_..._az_....fits

All IRF files found under that one NSB directory are handed to
``lstchain_create_dl3_file`` at once (via ``--irf-file-pattern``), which
interpolates over them in the pointing-direction space (one file per
simulated pointing node); no declination-based selection is done.

For each selected real DL2 file, a DL3 FITS file is created with:

    lstchain_create_dl3_file
        --input-dl2 <file>
        --output-dl3-path <output-dl3-dir>
        --input-irf-path <irf-base-dir>/fits/<mc-tag>_nsb_<NSB>
        --irf-file-pattern <pattern>
        --source-name <source-name>
        --config <config>
        [--overwrite] [--source-dep] [--use-nearest-irf-node] [--gzip] ...

With --sbatch, each DL3 creation is instead submitted as its own SLURM batch
job (via a generated script under <output-dl3-dir>/job_scripts/ and `sbatch`),
so that files are processed in parallel across the cluster instead of one at
a time.

Example
-------
run_create_dl3_files.py \\
    --config /path/to/dl3_tool_config.json \\
    --mc-tag 20250212_v0.10.17_allsky_interp_dl2_irfs \\
    --nsb 0.14 0.22 0.38 0.50 \\
    --target "Crab Nebula" \\
    --input-dl2-base-dir /path/to/real/DL2/CrabNebula \\
    --irf-base-dir /path/to/output/full_enclosure/intens200_leak0.2_gheff0.6 \\
    --output-dl3-dir /path/to/output/DL3/CrabNebula

Or, with an explicit, already-selected list of DL2 files (e.g. a good-run
file list built elsewhere) instead of --input-dl2-base-dir:

run_create_dl3_files.py \\
    --config /path/to/dl3_tool_config.json \\
    --nsb 0.14 0.22 0.38 0.50 \\
    --target "Crab Nebula" \\
    --input-dl2 /path/to/dl2_LST-1.Run00001.h5 /path/to/dl2_LST-1.Run00002.h5 ... \\
    --irf-base-dir /path/to/output/full_enclosure/intens200_leak0.2_gheff0.6 \\
    --output-dl3-dir /path/to/output/DL3/CrabNebula \\
    --srun --srun-args --mem 10g
"""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

from lstchain.paths import dl2_to_dl3_filename

log = logging.getLogger(__name__)

# Subdirectory of --output-dl3-dir holding the SLURM batch scripts generated
# when --sbatch is used.
JOB_SCRIPTS_SUBDIR = "job_scripts"

# Top-level subdirectory of --irf-base-dir holding the IRF FITS files.
IRF_FITS_SUBDIR = "fits"

# Characters allowed unescaped in a SLURM job name; anything else is replaced
# with "_" (job names are also used as the batch script's file name).
_JOB_TAG_RE = re.compile(r"[^A-Za-z0-9_.-]")

# Default glob pattern (relative to --input-dl2-base-dir) for real DL2 data
# files at a given NSB tuning level; "{nsb}" is replaced with each --nsb
# value formatted as "%.2f".
DEFAULT_DL2_GLOB = "**/nsb_tuning_{nsb:.2f}/dl2_LST-1.Run*.h5"

# Default substring (relative to --input-dl2) used to pick out, from an
# explicit list of DL2 files passed with --input-dl2, the ones belonging to
# a given NSB tuning level; "{nsb}" is replaced with each --nsb value
# formatted as "%.2f".
DEFAULT_DL2_NSB_TAG = "/nsb_tuning_{nsb:.2f}/"


def find_irf_dir(irf_base_dir, mc_tag, nsb_val):
    """Return the IRF directory to use for one NSB level:
    <irf-base-dir>/fits/<mc-tag>_nsb_<NSB>/. IRF files for all pointing
    nodes (and, if present, several declinations) are expected directly
    inside this one directory, with no further subdirectories -- selecting
    among them (by declination or otherwise) is left entirely to
    lstchain_create_dl3_file's own IRF interpolation.
    """
    irf_dir = irf_base_dir / IRF_FITS_SUBDIR / f"{mc_tag}_nsb_{nsb_val:.2f}"
    if not irf_dir.is_dir():
        raise FileNotFoundError(f"IRF directory not found: {irf_dir}")
    return irf_dir


def find_dl2_files(base_dir, glob_pattern, nsb_val):
    """List real DL2 data files for one NSB level under base_dir."""
    pattern = glob_pattern.format(nsb=nsb_val)
    files = sorted(base_dir.glob(pattern))
    if not files:
        log.warning(
            "No DL2 files found under %s (pattern: %s)", base_dir, pattern
        )
    else:
        log.info("Found %d DL2 file(s) for NSB=%.2f under %s", len(files), nsb_val, base_dir)
    return files


def select_dl2_files_by_tag(files, nsb_tag_template, nsb_val):
    """Pick, from an explicit list of DL2 files (--input-dl2), the ones
    belonging to one NSB level: those whose path contains
    nsb_tag_template.format(nsb=nsb_val), e.g. "/nsb_tuning_0.14/".
    """
    tag = nsb_tag_template.format(nsb=nsb_val)
    selected = sorted(f for f in files if tag in str(f))
    if not selected:
        log.warning(
            "No DL2 files matching tag '%s' found in --input-dl2 for NSB=%.2f", tag, nsb_val
        )
    else:
        log.info("Selected %d DL2 file(s) for NSB=%.2f (tag: '%s')", len(selected), nsb_val, tag)
    return selected


def build_dl3_command(infile, irf_dir, args):
    """Build the lstchain_create_dl3_file command line (as a list of tokens,
    without any 'srun'/'sbatch' wrapper) for one input real DL2 file.
    """
    cmd = [
        "lstchain_create_dl3_file",
        "--input-dl2", str(infile),
        "--output-dl3-path", str(args.output_dl3_dir),
        "--input-irf-path", str(irf_dir),
        "--irf-file-pattern", args.irf_file_pattern,
        "--source-name", args.source_name,
        "--interp-method", args.interp_method,
        "--config", str(args.config),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.source_dep:
        cmd.append("--source-dep")
    if args.use_nearest_irf_node:
        cmd.append("--use-nearest-irf-node")
    if args.gzip:
        cmd.append("--gzip")
    cmd += args.extra_args
    return cmd


def run_lstchain_create_dl3_file(infile, irf_dir, args):
    """Run lstchain_create_dl3_file directly (blocking) for one input real
    DL2 file. Files are processed one at a time; see
    submit_lstchain_create_dl3_file_job (--sbatch) for parallel processing.
    """
    args.output_dl3_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_dl3_command(infile, irf_dir, args)
    if args.srun:
        cmd = ["srun"] + args.srun_args + cmd

    log.info("Running: %s", " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def submit_lstchain_create_dl3_file_job(infile, irf_dir, job_scripts_dir, args, job_tag):
    """Write a SLURM batch script running lstchain_create_dl3_file for one
    input real DL2 file, and submit it with sbatch, so that many files can
    be processed in parallel across the cluster instead of one at a time.

    ``job_tag`` must be unique across the whole run (e.g. combining NSB and
    the input file's run number) since it is used both as the SLURM job
    name and as the batch script's file name.

    Returns True if the job was submitted (or would be, under --dry-run).
    """
    args.output_dl3_dir.mkdir(parents=True, exist_ok=True)
    job_scripts_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_dl3_command(infile, irf_dir, args)
    if args.srun:
        cmd = ["srun"] + args.srun_args + cmd

    job_name = "dl3_" + _JOB_TAG_RE.sub("_", job_tag)[:56]

    script_lines = [
        "#!/bin/sh",
        f"#SBATCH -p {args.sbatch_partition}",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -o {job_scripts_dir}/{job_name}-%j.out",
        f"#SBATCH --mem={args.sbatch_mem}",
        "#SBATCH -N 1",
    ]
    if args.sbatch_exclusive:
        script_lines.append("#SBATCH --exclusive")
    script_lines += [
        "",
        "ulimit -l unlimited",
        "ulimit -s unlimited",
        "ulimit -a",
        "",
        " ".join(cmd),
        "",
    ]

    job_script_path = job_scripts_dir / f"{job_name}.sh"
    job_script_path.write_text("\n".join(script_lines))

    command_elements = ["sbatch", str(job_script_path)]
    log.info("Submitting: %s", " ".join(command_elements))
    if args.dry_run:
        return True
    try:
        subprocess.run(command_elements, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        log.error(
            "sbatch submission failed for %s (return code %d)\nstdout: %s\nstderr: %s",
            job_script_path, e.returncode, e.stdout, e.stderr,
        )
        return False


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create LST-1 DL3 FITS files with lstchain_create_dl3_file from real DL2 "
            "data files for one or more NSB levels, using all the IRF FITS files found "
            "under each NSB level's IRF directory."
        ),
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to the lstchain_create_dl3_file JSON config file (passed through "
             "as --config).",
    )
    parser.add_argument(
        "--target", required=True,
        help='Target name, e.g. "Crab Nebula". Used as the default --source-name if '
             "--source-name is not given; lstchain_create_dl3_file resolves it to sky "
             "coordinates itself (via astropy SkyCoord.from_name) unless --source-ra/"
             "--source-dec are used instead (not exposed here; pass them via --extra-args).",
    )
    parser.add_argument(
        "--source-name", default=None,
        help="Source name passed through as --source-name to lstchain_create_dl3_file "
             "(default: same as --target).",
    )
    parser.add_argument(
        "--nsb", dest="nsb_values", type=float, nargs="+", required=True,
        help="One or more NSB values, e.g. --nsb 0.14 0.22 0.38 0.50.",
    )
    parser.add_argument(
        "--input-dl2-base-dir", type=Path, default=None,
        help="Base directory to search for real DL2 data files (mutually exclusive "
             "with --input-dl2). One of the two is required.",
    )
    parser.add_argument(
        "--input-dl2-glob", default=DEFAULT_DL2_GLOB,
        help="Glob pattern (relative to --input-dl2-base-dir) matching real DL2 data "
             "files for one NSB level; '{nsb}' is replaced with each --nsb value "
             "formatted as '%%.2f' (default: %(default)s)",
    )
    parser.add_argument(
        "--input-dl2", type=Path, nargs="+", default=None,
        help="Explicit list of real DL2 data files to process, e.g. an "
             "already quality-filtered good-run file list (mutually exclusive with "
             "--input-dl2-base-dir). One of the two is required. Each --nsb value's "
             "files are picked out of this list with --input-dl2-nsb-tag.",
    )
    parser.add_argument(
        "--input-dl2-nsb-tag", default=DEFAULT_DL2_NSB_TAG,
        help="Substring matched against each --input-dl2 file's path to select the "
             "files belonging to one NSB level; '{nsb}' is replaced with each --nsb "
             "value formatted as '%%.2f' (default: %(default)s)",
    )
    parser.add_argument(
        "--irf-base-dir", type=Path, required=True,
        help=f"Base directory containing '{IRF_FITS_SUBDIR}/<mc-tag>_nsb_<NSB>/' "
             "IRF FITS files (placed directly in that directory, with no further "
             "declination or other subdirectory).",
    )
    parser.add_argument(
        "--mc-tag", default="20250212_v0.10.17_allsky_interp_dl2_irfs",
        help="DL2 MC directory tag, matching the one used with run_create_irf_files.py "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--irf-file-pattern", default="irf*.fits",
        help="Glob pattern (passed through as --irf-file-pattern) selecting IRF FITS "
             "files within each NSB level's IRF directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dl3-dir", type=Path, required=True,
        help="Output directory for DL3 FITS files, passed through as --output-dl3-path. "
             f"SLURM batch scripts (--sbatch) are written under its {JOB_SCRIPTS_SUBDIR}/ "
             "subdirectory.",
    )
    parser.add_argument(
        "--interp-method", default="linear",
        help="IRF interpolation method, passed through as --interp-method (default: "
             "%(default)s)",
    )
    parser.add_argument(
        "--use-nearest-irf-node", action="store_true",
        help="Pass --use-nearest-irf-node: use only the nearest IRF node to the data "
             "in the interpolation space (pointing direction), instead of interpolating.",
    )
    parser.add_argument(
        "--source-dep", action="store_true",
        help="Pass --source-dep: perform source-dependent analysis.",
    )
    parser.add_argument(
        "--gzip", action="store_true",
        help="Pass --gzip: gzip the output DL3 files.",
    )
    parser.add_argument(
        "--no-overwrite", dest="overwrite", action="store_false",
        help="Do not pass --overwrite to lstchain_create_dl3_file (passed by default)",
    )
    parser.add_argument(
        "--srun", action="store_true",
        help="Prepend 'srun' to the lstchain_create_dl3_file command line (SLURM clusters)",
    )
    parser.add_argument(
        "--srun-args", nargs="+", default=[],
        help="Extra arguments inserted right after 'srun' when --srun is used, e.g. "
             "--srun-args --mem 10g (default: none)",
    )
    parser.add_argument(
        "--sbatch", action="store_true",
        help="Submit each DL3 creation as its own SLURM batch job (via sbatch) instead "
             "of running lstchain_create_dl3_file directly, so files are processed in "
             "parallel across the cluster. Job scripts are written under "
             f"<output-dl3-dir>/{JOB_SCRIPTS_SUBDIR}/. The script does not wait for the "
             "jobs to finish.",
    )
    parser.add_argument(
        "--sbatch-partition", default="short",
        help="SLURM partition for --sbatch jobs (default: %(default)s)",
    )
    parser.add_argument(
        "--sbatch-mem", default="10g",
        help="SLURM --mem value for --sbatch jobs (default: %(default)s)",
    )
    parser.add_argument(
        "--no-sbatch-exclusive", dest="sbatch_exclusive", action="store_false",
        help="Do not request exclusive node access (--exclusive) for --sbatch jobs "
             "(requested by default)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would be run, without executing them",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s)",
    )
    parser.add_argument(
        "--extra-args", nargs=argparse.REMAINDER, default=[],
        help="Additional arguments appended verbatim to the lstchain_create_dl3_file "
             "command line. Must be given last, after all other options.",
    )
    parser.set_defaults(overwrite=True, sbatch_exclusive=True)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if args.source_name is None:
        args.source_name = args.target

    if (args.input_dl2 is None) == (args.input_dl2_base_dir is None):
        parser.error(
            "exactly one of --input-dl2 or --input-dl2-base-dir must be given"
        )

    if not args.config.is_file() and not args.dry_run:
        log.error("Config file not found: %s", args.config)
        sys.exit(1)

    job_scripts_dir = args.output_dl3_dir / JOB_SCRIPTS_SUBDIR
    args.output_dl3_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Run configuration: target='%s', source_name=%s, config=%s, "
        "input_dl2=%s, input_dl2_base_dir=%s, irf_base_dir=%s, mc_tag=%s, nsb=%s, "
        "overwrite=%s, srun=%s, sbatch=%s",
        args.target, args.source_name, args.config,
        f"{len(args.input_dl2)} file(s)" if args.input_dl2 else None,
        args.input_dl2_base_dir, args.irf_base_dir, args.mc_tag, args.nsb_values,
        args.overwrite, args.srun, args.sbatch,
    )
    if args.sbatch:
        log.info(
            "sbatch job settings: partition=%s, mem=%s, exclusive=%s, job_scripts_dir=%s",
            args.sbatch_partition, args.sbatch_mem, args.sbatch_exclusive, job_scripts_dir,
        )
    log.info("DL3 FITS files will be written under: %s", args.output_dl3_dir)

    n_run = 0
    n_submitted = 0
    n_skipped_existing = 0
    for nsb_val in args.nsb_values:
        if args.input_dl2 is not None:
            infiles = select_dl2_files_by_tag(args.input_dl2, args.input_dl2_nsb_tag, nsb_val)
        else:
            log.info("NSB=%.2f: scanning DL2 directory %s", nsb_val, args.input_dl2_base_dir)
            infiles = find_dl2_files(args.input_dl2_base_dir, args.input_dl2_glob, nsb_val)
        if not infiles:
            continue

        irf_dir = find_irf_dir(args.irf_base_dir, args.mc_tag, nsb_val)

        for infile in infiles:
            outfile = args.output_dl3_dir / dl2_to_dl3_filename(infile, compress=args.gzip)
            log.info("Input DL2 file: %s", infile)
            log.info("Output DL3 FITS file: %s", outfile)

            if outfile.exists() and not args.overwrite:
                log.info("Output file already exists: %s. Skipping.", outfile)
                n_skipped_existing += 1
                continue

            if args.sbatch:
                job_tag = f"nsb{nsb_val:.2f}_{outfile.stem}"
                if submit_lstchain_create_dl3_file_job(
                    infile, irf_dir, job_scripts_dir, args, job_tag
                ):
                    n_submitted += 1
                continue

            run_lstchain_create_dl3_file(infile, irf_dir, args)
            n_run += 1
            if not args.dry_run and not outfile.exists():
                log.error("Expected output file was not created: %s", outfile)

    if args.sbatch:
        log.info(
            "Done. Submitted %d sbatch job(s), %d output file(s) already existed and "
            "were skipped. Jobs run asynchronously.",
            n_submitted, n_skipped_existing,
        )
    else:
        log.info(
            "Done. Ran lstchain_create_dl3_file for %d file(s), %d output file(s) "
            "already existed and were skipped.",
            n_run, n_skipped_existing,
        )


if __name__ == "__main__":
    main()
