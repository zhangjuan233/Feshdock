# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
  Feshdock is developed based on modifications to the LightDock source code.
"""
import sys,os
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feshdock.util.logger import LoggingManager
from feshdock.util.parser import CommandLineParser


log = LoggingManager.get_logger("feshdock")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        parser = CommandLineParser()

        mpi_support = parser.args.mpi
        if mpi_support:
            from feshdock.simulation.docking_mpi import (
                run_simulation as mpi_simulation,
            )

            mpi_simulation(parser)
        else:
            from feshdock.simulation.docking_multiprocessing import (
                run_simulation as multiprocessing_simulation,
            )
            multiprocessing_simulation(parser)

    except Exception as e:
        log.error("feshdock has failed, please check traceback:")
        traceback.print_exc()
        raise e
