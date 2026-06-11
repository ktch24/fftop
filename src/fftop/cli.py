import argparse

from .topology import process_field_data
from .dave4vm import run_dave4vm_series
from .potential import main as potential_module_main
from .workflow import main as workflow_main


def topology_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FFTop topology processing on a Zarr field store."
    )
    parser.add_argument("--param-file", required=True, help="Path to specifications.txt")
    parser.add_argument("--field-loc", required=True, help="Path to the field_data_<tag>.zarr store, or a directory containing it")
    parser.add_argument("--field-tag", required=True, help="Field tag / SHARP number, e.g. 7237")
    parser.add_argument("--vel-tag", required=True, help="Velocity smoothing tag, e.g. 12 or 20")
    parser.add_argument("--savdir", required=True, help="Directory where topology outputs will be written")
    parser.add_argument("--zarr-file", default=None, help="Output topology zarr filename (default: topology_<field_tag>.zarr)")
    parser.add_argument("--ck", type=int, default=16, help="Chunk size in snapshots (default: 16)")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of worker processes (default: cpu_count()-1 inside code)")
    parser.add_argument("--dx", type=float, default=360.0, help="Grid spacing in x (default: 360.0)")
    parser.add_argument("--dy", type=float, default=360.0, help="Grid spacing in y (default: 360.0)")
    parser.add_argument("--dz", type=float, default=1.0, help="Grid spacing in z/time-slice direction for the topology code (default: 1.0)")
    args = parser.parse_args()

    zarr_file = args.zarr_file or f"topology_{args.field_tag}.zarr"

    process_field_data(
        param_file=args.param_file,
        ck=args.ck,
        field_loc=args.field_loc,
        field_tag=str(args.field_tag),
        vel_tag=str(args.vel_tag),
        savdir=args.savdir,
        zarr_file=zarr_file,
        steps=(args.dx, args.dy, args.dz),
        n_workers=args.n_workers,
    )


def dave4vm_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DAVE4VM on a Zarr field store."
    )
    parser.add_argument("field_loc", help="Path to field_data_<tag>.zarr or a directory containing it")
    parser.add_argument("field_tag", help="Field tag / SHARP number")
    parser.add_argument("nx", type=int, help="Grid size in x")
    parser.add_argument("ny", type=int, help="Grid size in y")
    parser.add_argument("start_index", type=int, help="Start snapshot index")
    parser.add_argument("end_index", type=int, help="End snapshot index")
    parser.add_argument("window", type=int, help="DAVE4VM window size")
    parser.add_argument("--dx", type=float, default=360.0)
    parser.add_argument("--dy", type=float, default=360.0)
    parser.add_argument("--dt-seconds", type=float, default=720.0)
    args = parser.parse_args()

    run_dave4vm_series(
        field_loc=args.field_loc,
        field_tag=str(args.field_tag),
        nx=args.nx,
        ny=args.ny,
        start_index=args.start_index,
        end_index=args.end_index,
        window=args.window,
        dx=args.dx,
        dy=args.dy,
        dt_seconds=args.dt_seconds,
    )


def run_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FFTop region-processing pipeline."
    )
    parser.add_argument(
        "--region-file",
        required=True,
        help="Path to the region processing file (formerly read_data.txt).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download/extraction stage and use existing prepared data.",
    )

    parser.add_argument(
        "--topology-only",
        action="store_true",
        help="Skip straight to the topology calculations",
    )

    parser.add_argument(
  	"--ck",
        type=int,
        default=16,
        help="Chunk size for topology (default: 16)",
    )

    args = parser.parse_args()

    workflow_main(
        args.region_file,
        skip_download=args.skip_download,
        topology_only=args.topology_only,
	ck=args.ck,
    )


def potential_main() -> None:
    potential_module_main()


def download_main() -> None:
    from .download import main
    main()
