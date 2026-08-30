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
relative path, but under a sibling "plots" directory instead of "fits":

    <output-dir>/plots/<mc-tag>_nsb_<NSB>/<dec-dir>/irfs_theta_..._az_..._irf_diagnostics.png
    <output-dir>/plots/<mc-tag>_nsb_<NSB>/<dec-dir>/irfs_theta_..._az_..._psf.png

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
# FITS files and their diagnostic PNG plots (mirrored directory structure).
FITS_SUBDIR = "fits"
PLOTS_SUBDIR = "plots"

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


def run_lstchain_create_irf_files(infile, outfile, args):
    """Run lstchain_create_irf_files for one input DL2 MC gamma file."""
    outfile.parent.mkdir(parents=True, exist_ok=True)

    cmd = []
    if args.srun:
        cmd.append("srun")
    cmd += [
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

    log.info("Running: %s", " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def make_diagnostic_plots(irf_file, plot_dir, energy_dependent_gh):
    """Create PNG plots summarizing the contents of one IRF FITS file.

    Two files are written under ``plot_dir``:
    ``<name>_irf_diagnostics.png`` (effective area, energy dispersion bias,
    gamma/hadron cut) and ``<name>_psf.png`` (point spread function).
    Creation time and this script's path are embedded as PNG metadata.
    Returns the list of PNG paths that were created.
    """
    # Imported lazily so that --skip-plots does not require gammapy/matplotlib.
    import matplotlib
    matplotlib.use("Agg")
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

    created = []
    try:
        aeff = EffectiveAreaTable2D.read(irf_file, hdu="EFFECTIVE AREA")
        edisp = EnergyDispersion2D.read(irf_file, hdu="ENERGY DISPERSION")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        aeff.plot(ax=axes[0, 0])
        axes[0, 0].set_title("Effective Area (energy-offset dependence)")

        aeff.plot_energy_dependence(ax=axes[0, 1], offset=[0.5 * u.deg])
        legend = axes[0, 1].get_legend()
        if legend is not None:
            legend.remove()
        axes[0, 1].set_yscale("log")
        axes[0, 1].grid(which="both", linestyle=":")
        axes[0, 1].set_xlim(0.01, 200)
        axes[0, 1].set_title("Effective Area (energy dependence, offset=0.5 deg)")

        edisp.plot_bias(ax=axes[1, 0], offset=[0.4 * u.deg], add_cbar=True)
        axes[1, 0].set_title("Energy Bias (offset=0.4 deg)")

        if energy_dependent_gh:
            try:
                gh_cut = QTable.read(irf_file, hdu="GH_CUTS")
                axes[1, 1].errorbar(
                    gh_cut["center"], gh_cut["cut"],
                    xerr=(
                        gh_cut["center"] - gh_cut["low"],
                        gh_cut["high"] - gh_cut["center"],
                    ),
                )
                axes[1, 1].set_xscale("log")
                axes[1, 1].set_title(r"$\gamma$/h cut")
                axes[1, 1].set_ylabel(r"$\gamma$/h cut")
                axes[1, 1].set_xlabel("Energy [TeV]")
                axes[1, 1].grid(which="both")
            except KeyError:
                axes[1, 1].axis("off")
        else:
            axes[1, 1].axis("off")

        fig.suptitle(irf_file.name)
        fig.tight_layout()

        diagnostics_png = plot_dir / f"{base_name}_irf_diagnostics.png"
        fig.savefig(diagnostics_png, dpi=150, metadata=metadata)
        plt.close(fig)
        created.append(diagnostics_png)
        log.info("Wrote %s", diagnostics_png)

        psf = PSF3D.read(irf_file, hdu="PSF")
        psf.peek()
        psf_fig = plt.gcf()
        psf_fig.suptitle(irf_file.name)
        psf_png = plot_dir / f"{base_name}_psf.png"
        psf_fig.savefig(psf_png, dpi=150, metadata=metadata)
        plt.close(psf_fig)
        created.append(psf_png)
        log.info("Wrote %s", psf_png)
    except Exception:
        log.exception("Failed to create diagnostic plots for %s", irf_file)

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
        "--config", required=True, type=Path,
        help="Path to the lstchain_create_irf_files JSON config file "
             "(passed through as --config)",
    )
    parser.add_argument(
        "--target", required=True,
        help='Target name resolved with astropy SkyCoord.from_name, e.g. "Crab Nebula"',
    )
    parser.add_argument(
        "--nsb", dest="nsb_values", required=True, type=float, nargs="+",
        help="One or more NSB values, e.g. --nsb 0.14 0.22 0.38 0.50",
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
        "--output-dir", required=True, type=Path,
        help="Base output directory. IRF FITS files are written under "
             f"<output-dir>/{FITS_SUBDIR}/... and diagnostic plots under "
             f"<output-dir>/{PLOTS_SUBDIR}/..., mirroring the same subdirectory layout",
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
    parser.set_defaults(energy_dependent_gh=True, overwrite=True)
    return parser


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if not args.config.is_file() and not args.dry_run:
        log.error("Config file not found: %s", args.config)
        sys.exit(1)

    target_dec = resolve_target_dec(args.target)
    fits_root = args.output_dir / FITS_SUBDIR
    plots_root = args.output_dir / PLOTS_SUBDIR
    fits_root.mkdir(parents=True, exist_ok=True)

    log.info(
        "Run configuration: target='%s' (dec=%.4f deg), config=%s, "
        "mc_base_dir=%s, mc_tag=%s, dataset=%s, particle=%s, nsb=%s, "
        "energy_dependent_gh=%s, overwrite=%s, srun=%s",
        args.target, target_dec, args.config,
        args.mc_base_dir, args.mc_tag, args.dataset, args.particle, args.nsb_values,
        args.energy_dependent_gh, args.overwrite, args.srun,
    )
    log.info("IRF FITS files will be written under: %s", fits_root)
    log.info("Diagnostic plots will be written under: %s", plots_root)

    created_irf_files = []
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
                run_lstchain_create_irf_files(infile, outfile, args)
                if args.dry_run:
                    continue
                if outfile.exists():
                    log.info("Created IRF FITS file: %s", outfile)
                    created_irf_files.append((outfile, plots_root / rel_path.parent))
                else:
                    log.error("Expected output file was not created: %s", outfile)

    created_pngs = []
    if not args.skip_plots and not args.dry_run:
        for irf_file, plot_dir in created_irf_files:
            log.info("Creating diagnostic plots for %s in %s", irf_file, plot_dir)
            created_pngs += make_diagnostic_plots(irf_file, plot_dir, args.energy_dependent_gh)

    log.info(
        "Done. Created %d IRF FITS file(s) and %d diagnostic PNG file(s).",
        len(created_irf_files), len(created_pngs),
    )


if __name__ == "__main__":
    main()
