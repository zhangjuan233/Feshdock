# -*- coding: utf-8 -*-
"""
  Feshdock is developed based on modifications to the LightDock source code.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feshdock.util.parser import SetupCommandLineParser
from feshdock.prep.simulation import (
    read_input_structure,
    calculate_starting_positions,
    prepare_results_environment_2,
    create_setup_file,
    calculate_anm,
    parse_restraints_file,
    get_restraints,
    save_lightdock_structure,

)
from feshdock.constants import (
    DEFAULT_REC_NM_FILE,
    DEFAULT_LIG_NM_FILE,
    DEFAULT_STARTING_PREFIX
)
from feshdock.util.logger import LoggingManager
from feshdock.error.feshdock_errors import FeshdockError


log = LoggingManager.get_logger("feshdock_setup")


if __name__ == "__main__":
    try:
        parser = SetupCommandLineParser()
        args = parser.args

        # Read input structures
        receptor = read_input_structure(
            args.receptor_pdb, args.noxt, args.noh, args.now, args.verbose_parser
        )
        ligand = read_input_structure(
            args.ligand_pdb, args.noxt, args.noh, args.now, args.verbose_parser
        )
        # Move structures to origin
        rec_translation = receptor.move_to_origin()
        lig_translation = ligand.move_to_origin()

        # Save to file parsed structures
        save_lightdock_structure(receptor)
        save_lightdock_structure(ligand)


        # Calculate and save ANM if required
        if args.use_anm:
            if args.anm_rec > 0:
                log.info("Calculating ANM for receptor molecule...")
                calculate_anm(receptor, args.anm_rec, args.anm_rec_rmsd, args.anm_seed, DEFAULT_REC_NM_FILE)
            if args.anm_lig > 0:
                log.info("Calculating ANM for ligand molecule...")
                calculate_anm(ligand, args.anm_lig, args.anm_lig_rmsd, args.anm_seed, DEFAULT_LIG_NM_FILE)

        # Parse restraints if any:
        receptor_restraints = ligand_restraints = None
        if args.restraints:
            log.info(f"Reading restraints from {args.restraints}")
            restraints = parse_restraints_file(args.restraints)

            # Calculate number of restraints in order to check them
            num_rec_active = len(restraints["receptor"]["active"])
            num_rec_passive = len(restraints["receptor"]["passive"])
            num_rec_blocked = len(restraints["receptor"]["blocked"])
            num_lig_active = len(restraints["ligand"]["active"])
            num_lig_passive = len(restraints["ligand"]["passive"])

            # Complain if not a single restraint has been defined, but restraints are enabled
            if (
                not num_rec_active
                and not num_rec_passive
                and not num_rec_blocked
                and not num_lig_active
                and not num_lig_passive
            ):
                raise FeshdockError(
                    "Restraints file specified, but not a single restraint found"
                )

            # Check if restraints correspond with real residues
            receptor_restraints = get_restraints(receptor, restraints["receptor"])
            args.receptor_restraints = restraints["receptor"]
            ligand_restraints = get_restraints(ligand, restraints["ligand"])
            args.ligand_restraints = restraints["ligand"]

            log.info(
                f"Number of receptor restraints is: {num_rec_active} (active), {num_rec_passive} (passive)"
            )
            log.info(
                f"Number of ligand restraints is: {num_lig_active} (active), {num_lig_passive} (passive)"
            )

        try:
            lig_restraints = ligand_restraints["active"] + ligand_restraints["passive"]
        except (KeyError, TypeError):
            lig_restraints = None

        # swarm文件夹
        prepare_results_environment_2(args.swarms)
        create_setup_file(args)


    except FeshdockError as error:
        log.error("feshdock setup failed. Please see:")
        log.error(error)
