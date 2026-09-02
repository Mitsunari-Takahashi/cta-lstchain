#!/usr/bin/env python

"""
Batch-run ``lstchain_create_irf_files`` over LST-1 AllSky DL2 Monte-Carlo
gamma-diffuse files for one or more NSB levels, and produce diagnostic PNG
plots for every generated IRF FITS file.

This script is based on the workflow developed in the notebook
``DL3-analysis/notebooks/20260828_create_irfs.ipynb`` (LabNotebook_CTA repo).
The main difference is the declination-node selection: instead of using only
the single AllSky MC declination node closest to the target, this script uses
*both* the closest node above and the closest node below the target's
declination (so that, e.g., IRFs can later be interpolated in declination).

The AllSky MC production is assumed to be laid out as:

    <mc-base-dir>/<mc-tag>_nsb_<NSB>/<dataset>/<particle>/dec_<code>/
        node_corsika_theta_*_az_*_/*.h5

where a declination-node directory name such as ``dec_2276`` encodes a
declination of +22.76 deg, and ``dec_min_2276`` encodes -22.76 deg (the
integer is the declination in degrees, times 100). If your MC production uses
a different naming convention, adjust ``parse_dec_dir_name`` below.

For each selected input DL2 gamma file, an IRF FITS file is created with:

    lstchain_create_irf_files
        --input-gamma-dl2 <file>
        --output-irf-file <output-dir>/fits/<mc-tag>_nsb_<NSB>/<dec-dir>/irfs_theta_..._az_....fits
        --config <config>
        [--energy-dependent-gh] [--overwrite] ...

The diagnostic PNG plots for each IRF file are written under the same
relative path, but under a sibling "plots" directory instead of "fits", one
PNG per IRF component (effective area, energy dispersion & bias,
gamma/hadron cut, PSF):

    <output-dir>/plots/<mc-tag>_nsb_<NSB>/<dec-dir>/irfs_theta_..._az_..._aeff.png
    <output-dir>/plots/<mc-tag>_nsb_<NSB>/<dec-dir>/irfs_theta_..._az_..._edisp.png
    <output-dir>/plots/<mc-tag>_nsb_<NSB>/<dec-dir>/irfs_theta_..._az_..._gh_cut.png
    <output-dir>/plots/<mc-tag>_nsb_<NSB>/<dec-dir>/irfs_theta_..._az_..._psf.png

With --sbatch, each IRF creation is instead submitted as its own SLURM batch
job (via a generated script under <output-dir>/job_scripts/ and `sbatch`),
so that files are processed in parallel across the cluster instead of one
at a time. Each job is self-contained: unless --skip-plots is given, it
also generates its own diagnostic plots right after creating its IRF file,
by re-invoking this script in --plot-only mode. (Outputs that already
existed before this run, and were therefore not resubmitted, are instead
plotted immediately by this script itself.)

Example
-------
run_create_irf_files.py \\
    --config /path/to/irf_tool_config.json \\
    --mc-tag 20250212_v0.10.17_allsky_interp_dl2_irfs \\
    --nsb 0.14 0.22 0.38 0.50 \\
    --target "Crab Nebula" \\
    --output-dir /path/to/output/full_enclosure/intens200_leak0.2_gheff0.6
"""

import argparse
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from astropy.coordinates import SkyCoord

log = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve()

# Top-level subdirectories of --output-dir holding, respectively, the IRF
# FITS files, their diagnostic PNG plots (mirrored directory structure), and
# the SLURM batch scripts generated when --sbatch is used.
FITS_SUBDIR = "fits"
PLOTS_SUBDIR = "plots"
JOB_SCRIPTS_SUBDIR = "job_scripts"

# Characters allowed unescaped in a SLURM job name; anything else is replaced
# with "_" (job names are also used as the batch script's file name).
_JOB_TAG_RE = re.compile(r"[^A-Za-z0-9_.-]")

# Matches AllSky MC declination-node directory names, e.g. "dec_2276" (+22.76
# deg) or "dec_min_2276" (-22.76 deg).
DEC_DIR_RE = re.compile(r"^dec_(min_)?(\d+)$")

# Glob pattern (relative to a dec_* directory) matching DL2 MC gamma-diffuse
# files, one per simulated (theta, az) pointing node.
NODE_FILE_GLOB = "node_corsika_theta_*_az_*_/*.h5"

# Extracts the "_theta<...>_az_<...>" pointing tag from a DL2 MC file name, to
# reuse it in the output IRF file name.
ANGLE_TAG_RE = re.compile(r"_theta.*?_az_[\d.]+")


def parse_dec_dir_name(name):
    """Convert a declination-node directory name into a declination in degrees.

    ``dec_2276`` -> 22.76, ``dec_min_2276`` -> -22.76.
    """
    m = DEC_DIR_RE.match(name)
    if not m:
        raise ValueError(
            f"'{name}' does not match the expected 'dec_<int>' / "
            "'dec_min_<int>' pattern"
        )
    sign = -1.0 if m.group(1) else 1.0
    return sign * int(m.group(2)) / 100.0


def resolve_target_dec(target_name):
    """Resolve an astronomical target name to a declination in degrees."""
    coord = SkyCoord.from_name(target_name)
    log.info(
        "Resolved target '%s' to RA=%.4f deg, Dec=%.4f deg",
        target_name, coord.ra.deg, coord.dec.deg,
    )
    return coord.dec.deg


def find_dec_dirs(base_dir):
    """Return {declination_deg: Path} for every dec_* subdirectory of base_dir."""
    if not base_dir.is_dir():
        raise FileNotFoundError(f"MC directory not found: {base_dir}")

    dec_dirs = {}
    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        try:
            dec_dirs[parse_dec_dir_name(path.name)] = path
        except ValueError:
            continue

    if not dec_dirs:
        raise FileNotFoundError(f"No 'dec_*' directories found under {base_dir}")
    return dec_dirs


def select_bracketing_decs(dec_dirs, target_dec):
    """Pick the closest available declination node below and above target_dec.

    Returns a sorted list with one or two declination values (in degrees).
    Only one value is returned if the target lies outside the range of
    available MC declination nodes (extrapolation, not interpolation).
    """
    decs = sorted(dec_dirs)
    below = [d for d in decs if d <= target_dec]
    above = [d for d in decs if d >= target_dec]

    selected = []
    if below:
        selected.append(max(below))
    if above:
        closest_above = min(above)
        if closest_above not in selected:
            selected.append(closest_above)

    if not selected:
        raise RuntimeError(
            f"No declination nodes available to bracket target dec={target_dec} deg"
        )
    if len(selected) == 1:
        log.warning(
            "Target dec=%.3f deg is outside the range of available MC dec nodes "
            "(%.3f to %.3f deg); using only the nearest node dec=%.3f deg",
            target_dec, decs[0], decs[-1], selected[0],
        )
    return sorted(selected)


def list_node_files(dec_dir):
    """List DL2 MC gamma-diffuse files under a declination-node directory."""
    files = sorted(dec_dir.glob(NODE_FILE_GLOB))
    if not files:
        log.warning("No MC files found under %s (pattern: %s)", dec_dir, NODE_FILE_GLOB)
    else:
        log.info("Found %d MC file(s) under %s", len(files), dec_dir)
    return files


def build_relative_output_path(mc_tag, nsb_val, dec_dir_name, infile):
    """Build the IRF FITS file path for one input DL2 MC file, relative to
    the "fits" (or "plots") root, i.e. <mc-tag>_nsb_<NSB>/<dec-dir>/<name>.
    """
    m = ANGLE_TAG_RE.search(infile.name)
    if not m:
        raise ValueError(f"Could not parse theta/az pointing tag from: {infile.name}")
    angle_tag = m.group(0)
    outname = f"irfs{angle_tag}.fits"
    return Path(f"{mc_tag}_nsb_{nsb_val:.2f}") / dec_dir_name / outname


def build_irf_command(infile, outfile, args):
    """Build the lstchain_create_irf_files command line (as a list of tokens,
    without any 'srun'/'sbatch' wrapper) for one input DL2 MC gamma file.
    """
    cmd = [
        "lstchain_create_irf_files",
        "--input-gamma-dl2", str(infile),
        "--output-irf-file", str(outfile),
        "--config", str(args.config),
    ]
    if args.input_proton_dl2:
        cmd += ["--input-proton-dl2", str(args.input_proton_dl2)]
    if args.input_electron_dl2:
        cmd += ["--input-electron-dl2", str(args.input_electron_dl2)]
    if args.energy_dependent_gh:
        cmd.append("--energy-dependent-gh")
    if args.overwrite:
        cmd.append("--overwrite")
    cmd += args.extra_args
    return cmd


def run_lstchain_create_irf_files(infile, outfile, args):
    """Run lstchain_create_irf_files directly (blocking) for one input DL2
    MC gamma file. Files are processed one at a time; see
    submit_lstchain_create_irf_files_job (--sbatch) for parallel processing.
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_irf_command(infile, outfile, args)
    if args.srun:
        cmd = ["srun"] + cmd

    log.info("Running: %s", " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def submit_lstchain_create_irf_files_job(infile, outfile, plot_dir, job_scripts_dir, args, job_tag):
    """Write a SLURM batch script running lstchain_create_irf_files for one
    input DL2 MC gamma file, and submit it with sbatch, so that many files
    can be processed in parallel across the cluster instead of one at a time.

    Unless ``args.skip_plots`` is set, the job script also generates that
    file's diagnostic plots (into ``plot_dir``) right after creating it, by
    re-invoking this same script in --plot-only mode -- so each job is fully
    self-contained (one IRF file in, its FITS file and PNGs out) and no
    separate plotting pass is needed once the jobs finish.

    ``job_tag`` must be unique across the whole run (e.g. combining NSB,
    declination node and pointing angle) since it is used both as the SLURM
    job name and as the batch script's file name.

    Returns True if the job was submitted (or would be, under --dry-run).
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)
    job_scripts_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_irf_command(infile, outfile, args)
    if args.srun:
        cmd = ["srun"] + cmd

    job_name = "irf_" + _JOB_TAG_RE.sub("_", job_tag)[:56]

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
        # Stop after the IRF creation step fails, so a broken/missing FITS
        # file is never handed to the plotting step below.
        "set -e",
        "",
        "ulimit -l unlimited",
        "ulimit -s unlimited",
        "ulimit -a",
        "",
        " ".join(cmd),
    ]
    if not args.skip_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--plot-only-file", str(outfile),
            "--plot-only-dir", str(plot_dir),
        ]
        if not args.energy_dependent_gh:
            plot_cmd.append("--no-energy-dependent-gh")
        script_lines.append(" ".join(plot_cmd))
    script_lines.append("")

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


def make_diagnostic_plots(irf_file, plot_dir, energy_dependent_gh):
    """Create PNG plots summarizing the contents of one IRF FITS file.

    Up to four files are written under ``plot_dir``, one per IRF component:
    ``<name>_aeff.png`` (effective area), ``<name>_edisp.png`` (energy
    dispersion and bias), ``<name>_gh_cut.png`` (gamma/hadron cut, only if
    ``energy_dependent_gh`` and a GH_CUTS HDU is present), and
    ``<name>_psf.png`` (point spread function). Creation time and this
    script's path are embedded as PNG metadata. Returns the list of PNG
    paths that were created; a failure on one plot does not prevent the
    others from being attempted.
    """
    # Imported lazily so that --skip-plots does not require gammapy/matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import astropy.units as u
    from astropy.table import QTable
    from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D, PSF3D
    from matplotlib import pyplot as plt


    base_name = irf_file.name
    for suffix in (".fits.gz", ".fits"):
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break

    metadata = {
        "Creation Time": datetime.now().astimezone().isoformat(),
        "Software": str(SCRIPT_PATH),
        "Source": str(irf_file),
    }

    plot_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, suffix):
        fig.suptitle(irf_file.name)
        fig.tight_layout()
        png_path = plot_dir / f"{base_name}_{suffix}.png"
        fig.savefig(png_path, dpi=150, metadata=metadata)
        plt.close(fig)
        log.info("Wrote %s", png_path)
        return png_path

    created = []

    # Effective Area: energy-offset dependence, energy dependence, offset dependence.
    try:
        aeff = EffectiveAreaTable2D.read(irf_file, hdu="EFFECTIVE AREA")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Effective area can be exactly 0 in some energy-offset bins (e.g.
        # below threshold), which is invalid for a log-scaled color norm.
        # Compute vmin/vmax ourselves (restricting vmin to positive values)
        # and pass them explicitly so gammapy's own `kwargs.setdefault(...)`
        # for vmin/vmax (using the raw, possibly-zero, nanmin) is not used.
        energy_true = aeff.axes["energy_true"]
        offset = aeff.axes["offset"]
        aeff_values = aeff.evaluate(
            offset=offset.center, energy_true=energy_true.center[:, np.newaxis]
        ).value
        positive_values = aeff_values[aeff_values > 0]
        vmin = positive_values.min() if positive_values.size else 1e-10
        vmax = aeff_values.max()

        aeff.plot(ax=axes[0], norm="log", vmin=vmin, vmax=vmax)
        axes[0].set_title("Energy-offset dependence")

        aeff.plot_energy_dependence(ax=axes[1], offset=[0.5 * u.deg])
        legend = axes[1].get_legend()
        if legend is not None:
            legend.remove()
        axes[1].set_yscale("log")
        axes[1].grid(which="both", linestyle=":")
        axes[1].set_xlim(0.01, 200)
        axes[1].set_title("Energy dependence (offset=0.5 deg)")

        aeff.plot_offset_dependence(ax=axes[2])
        axes[2].set_title("Offset dependence")

        created.append(save(fig, "aeff"))
    except Exception:
        log.exception("Failed to create effective area plot for %s", irf_file)

    # Energy Dispersion and Bias: gammapy's built-in 3-panel summary
    # (bias, migration distribution, dispersion matrix).
    try:
        edisp = EnergyDispersion2D.read(irf_file, hdu="ENERGY DISPERSION")
        edisp.peek()
        created.append(save(plt.gcf(), "edisp"))
    except Exception:
        log.exception("Failed to create energy dispersion plot for %s", irf_file)

    # Gamma/hadron cut: only meaningful for energy-dependent-gh IRFs.
    if energy_dependent_gh:
        try:
            gh_cut = QTable.read(irf_file, hdu="GH_CUTS")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.errorbar(
                gh_cut["center"], gh_cut["cut"],
                xerr=(
                    gh_cut["center"] - gh_cut["low"],
                    gh_cut["high"] - gh_cut["center"],
                ),
            )
            ax.set_xscale("log")
            ax.set_title(r"$\gamma$/h cut")
            ax.set_ylabel(r"$\gamma$/h cut")
            ax.set_xlabel("Energy [TeV]")
            ax.grid(which="both")
            created.append(save(fig, "gh_cut"))
        except Exception:
            log.exception("Failed to create gamma/hadron cut plot for %s", irf_file)

    # Point Spread Function: gammapy's built-in summary plot.
    try:
        psf = PSF3D.read(irf_file, hdu="PSF")
        psf.peek()
        created.append(save(plt.gcf(), "psf"))
    except Exception:
        log.exception("Failed to create PSF plot for %s", irf_file)

    return created


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create LST-1 IRF FITS files with lstchain_create_irf_files for one or "
            "more NSB levels, bracketing the target's declination with the closest "
            "available AllSky MC dec nodes above and below it, and plot diagnostics "
            "for each generated IRF file."
        ),
    )
    parser.add_argument(
        "--config", type=Path,
        help="Path to the lstchain_create_irf_files JSON config file "
             "(passed through as --config). Required unless --plot-only-file is used.",
    )
    parser.add_argument(
        "--target",
        help='Target name resolved with astropy SkyCoord.from_name, e.g. "Crab Nebula". '
             "Required unless --plot-only-file is used.",
    )
    parser.add_argument(
        "--nsb", dest="nsb_values", type=float, nargs="+",
        help="One or more NSB values, e.g. --nsb 0.14 0.22 0.38 0.50. "
             "Required unless --plot-only-file is used.",
    )
    parser.add_argument(
        "--mc-tag", default="20250212_v0.10.17_allsky_interp_dl2_irfs",
        help="DL2 MC directory tag, e.g. 20250212_v0.10.17_allsky_interp_dl2_irfs "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--mc-base-dir", type=Path, default=Path("/fefs/aswg/data/mc/DL2/AllSky"),
        help="Root directory of the AllSky DL2 MC production (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset", default="TestingDataset",
        help="Dataset subdirectory name under '<mc-tag>_nsb_<NSB>/' (default: %(default)s)",
    )
    parser.add_argument(
        "--particle", default="GammaDiffuse",
        help="Particle subdirectory name under the dataset directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Base output directory. IRF FITS files are written under "
             f"<output-dir>/{FITS_SUBDIR}/... and diagnostic plots under "
             f"<output-dir>/{PLOTS_SUBDIR}/..., mirroring the same subdirectory layout. "
             "Required unless --plot-only-file is used.",
    )
    parser.add_argument(
        "--plot-only-file", type=Path, default=None,
        help="Internal use: skip IRF creation entirely and just generate diagnostic "
             "plots (into --plot-only-dir) for this single, already-existing IRF FITS "
             "file, then exit. This is what --sbatch job scripts call on themselves "
             "after creating their IRF file; you normally don't need to pass this by "
             "hand, but it can also be used to (re)plot one file on its own.",
    )
    parser.add_argument(
        "--plot-only-dir", type=Path, default=None,
        help="Directory to write the PNGs into when --plot-only-file is used "
             "(required together with it).",
    )
    parser.add_argument(
        "--input-proton-dl2", type=Path, default=None,
        help="Optional MC proton DL2 file, passed through as --input-proton-dl2 "
             "(together with --input-electron-dl2, enables the BACKGROUND HDU)",
    )
    parser.add_argument(
        "--input-electron-dl2", type=Path, default=None,
        help="Optional MC electron DL2 file, passed through as --input-electron-dl2",
    )
    parser.add_argument(
        "--no-energy-dependent-gh", dest="energy_dependent_gh", action="store_false",
        help="Do not pass --energy-dependent-gh (it is passed by default)",
    )
    parser.add_argument(
        "--no-overwrite", dest="overwrite", action="store_false",
        help="Do not pass --overwrite to lstchain_create_irf_files (passed by default)",
    )
    parser.add_argument(
        "--srun", action="store_true",
        help="Prepend 'srun' to the lstchain_create_irf_files command line (SLURM clusters)",
    )
    parser.add_argument(
        "--sbatch", action="store_true",
        help="Submit each IRF creation as its own SLURM batch job (via sbatch) instead "
             "of running lstchain_create_irf_files directly, so files are processed in "
             "parallel across the cluster. Job scripts are written under "
             f"<output-dir>/{JOB_SCRIPTS_SUBDIR}/; each job also generates its own "
             "diagnostic plots right after creating its IRF file (unless --skip-plots "
             "is given). The script does not wait for the jobs to finish.",
    )
    parser.add_argument(
        "--sbatch-partition", default="short",
        help="SLURM partition for --sbatch jobs (default: %(default)s)",
    )
    parser.add_argument(
        "--sbatch-mem", default="5g",
        help="SLURM --mem value for --sbatch jobs (default: %(default)s)",
    )
    parser.add_argument(
        "--no-sbatch-exclusive", dest="sbatch_exclusive", action="store_false",
        help="Do not request exclusive node access (--exclusive) for --sbatch jobs "
             "(requested by default)",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Do not generate diagnostic PNG plots for the produced IRF files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would be run, without executing them or plotting",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s)",
    )
    parser.add_argument(
        "--extra-args", nargs=argparse.REMAINDER, default=[],
        help="Additional arguments appended verbatim to the lstchain_create_irf_files "
             "command line. Must be given last, after all other options.",
    )
    parser.set_defaults(energy_dependent_gh=True, overwrite=True, sbatch_exclusive=True)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if args.plot_only_file is not None or args.plot_only_dir is not None:
        if args.plot_only_file is None or args.plot_only_dir is None:
            parser.error("--plot-only-file and --plot-only-dir must be given together")
        created = make_diagnostic_plots(
            args.plot_only_file, args.plot_only_dir, args.energy_dependent_gh
        )
        log.info("Done. Created %d diagnostic PNG file(s).", len(created))
        return

    if args.config is None or args.target is None or not args.nsb_values or args.output_dir is None:
        parser.error("--config, --target, --nsb and --output-dir are required")

    if not args.config.is_file() and not args.dry_run:
        log.error("Config file not found: %s", args.config)
        sys.exit(1)

    target_dec = resolve_target_dec(args.target)
    fits_root = args.output_dir / FITS_SUBDIR
    plots_root = args.output_dir / PLOTS_SUBDIR
    job_scripts_dir = args.output_dir / JOB_SCRIPTS_SUBDIR
    fits_root.mkdir(parents=True, exist_ok=True)

    log.info(
        "Run configuration: target='%s' (dec=%.4f deg), config=%s, "
        "mc_base_dir=%s, mc_tag=%s, dataset=%s, particle=%s, nsb=%s, "
        "energy_dependent_gh=%s, overwrite=%s, srun=%s, sbatch=%s",
        args.target, target_dec, args.config,
        args.mc_base_dir, args.mc_tag, args.dataset, args.particle, args.nsb_values,
        args.energy_dependent_gh, args.overwrite, args.srun, args.sbatch,
    )
    if args.sbatch:
        log.info(
            "sbatch job settings: partition=%s, mem=%s, exclusive=%s, job_scripts_dir=%s",
            args.sbatch_partition, args.sbatch_mem, args.sbatch_exclusive, job_scripts_dir,
        )
    log.info("IRF FITS files will be written under: %s", fits_root)
    log.info("Diagnostic plots will be written under: %s", plots_root)

    created_irf_files = []  # (outfile, plot_dir) pairs ready to be plotted now
    n_submitted = 0
    n_skipped_existing = 0
    for nsb_val in args.nsb_values:
        particle_dir = (
            args.mc_base_dir
            / f"{args.mc_tag}_nsb_{nsb_val:.2f}"
            / args.dataset
            / args.particle
        )
        log.info("NSB=%.2f: scanning MC directory %s", nsb_val, particle_dir)
        dec_dirs = find_dec_dirs(particle_dir)
        log.info(
            "NSB=%.2f: available dec nodes: %s",
            nsb_val, {round(d, 3): dec_dirs[d].name for d in sorted(dec_dirs)},
        )
        selected_decs = select_bracketing_decs(dec_dirs, target_dec)
        log.info(
            "NSB=%.2f: bracketing dec nodes for target dec=%.3f deg: %s",
            nsb_val, target_dec,
            {round(d, 3): dec_dirs[d].name for d in selected_decs},
        )

        for dec_val in selected_decs:
            dec_dir = dec_dirs[dec_val]
            infiles = list_node_files(dec_dir)
            for infile in infiles:
                rel_path = build_relative_output_path(
                    args.mc_tag, nsb_val, dec_dir.name, infile
                )
                outfile = fits_root / rel_path
                log.info("Input DL2 MC gamma file: %s", infile)
                log.info("Output IRF FITS file: %s", outfile)

                if args.sbatch:
                    plot_dir = plots_root / rel_path.parent
                    if outfile.exists() and not args.overwrite:
                        log.info("Output file already exists: %s. Skipping.", outfile)
                        n_skipped_existing += 1
                        created_irf_files.append((outfile, plot_dir))
                        continue
                    job_tag = f"nsb{nsb_val:.2f}_{dec_dir.name}_{outfile.stem}"
                    if submit_lstchain_create_irf_files_job(
                        infile, outfile, plot_dir, job_scripts_dir, args, job_tag
                    ):
                        n_submitted += 1
                    continue

                run_lstchain_create_irf_files(infile, outfile, args)
                if args.dry_run:
                    continue
                if outfile.exists():
                    log.info("Created IRF FITS file: %s", outfile)
                    created_irf_files.append((outfile, plots_root / rel_path.parent))
                else:
                    log.error("Expected output file was not created: %s", outfile)

    if args.sbatch and not args.dry_run:
        plot_note = (
            "each job will also generate its own diagnostic plots after creating "
            "its IRF file" if not args.skip_plots else
            "diagnostic plots were not requested (--skip-plots)"
        )
        log.info(
            "Submitted %d sbatch job(s) (%d output file(s) already existed and were "
            "skipped). Jobs run asynchronously; %s.",
            n_submitted, n_skipped_existing, plot_note,
        )

    created_pngs = []
    if not args.skip_plots and not args.dry_run:
        for irf_file, plot_dir in created_irf_files:
            log.info("Creating diagnostic plots for %s in %s", irf_file, plot_dir)
            created_pngs += make_diagnostic_plots(irf_file, plot_dir, args.energy_dependent_gh)

    if args.sbatch:
        log.info(
            "Done. Submitted %d IRF job(s), %d already existed, %d diagnostic PNG "
            "file(s) created for pre-existing outputs.",
            n_submitted, n_skipped_existing, len(created_pngs),
        )
    else:
        log.info(
            "Done. Created %d IRF FITS file(s) and %d diagnostic PNG file(s).",
            len(created_irf_files), len(created_pngs),
        )


if __name__ == "__main__":
    main()
