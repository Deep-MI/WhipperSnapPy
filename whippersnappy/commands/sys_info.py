import argparse

from .. import sys_info


def run():
    """Run the sys_info command-line helper.

    Parses CLI arguments and delegates to the package-level `sys_info`
    function which prints system and dependency information.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function prints information to stdout via the `sys_info`
        helper and does not return a value.
    """
    parser = argparse.ArgumentParser(
        prog=f"{__package__.split('.')[0]}-sys_info", description="sys_info"
    )
    parser.add_argument(
        "--developer",
        help="display information for optional dependencies",
        action="store_true",
    )
    args = parser.parse_args()

    sys_info(developer=args.developer)
