import os
import math
import random
from feshdock.de.initializer import (
    RandomInitializer,
    FromFileInitializer,
    LightdockFromFileInitializer,
)


F0 = 0.45
class GSO(object):

    def __init__(
        self,
        swarm,
        gso_parameters,
        random_number_generator,
        initial_coordinates_file="",
        local_minimization=False,
    ):
        self.swarm = swarm
        self.parameters = gso_parameters
        self.random_number_generator = random_number_generator
        self.initial_coordinates_file = initial_coordinates_file
        self.local_minimization = local_minimization

    def run(
        self,
        simulation_steps,
        cluster_id=None,
        verbose=False,
        saving_path=".",
        save_intermediary=False,
        save_all_intermediary=False,
    ):
        """Runs the simulation for the given simulation_steps"""
        for i in range(len(self.swarm.glowworms)):
            self.swarm.glowworms[i].compute_scoring()
        self.swarm.my_sort()
        if save_intermediary:
            self.swarm.save(0, saving_path)

        for step in range(1, simulation_steps + 1):
            """  DE algorithm   """
            Gm = simulation_steps
            lamda = math.exp(1 - Gm / (Gm + 1 - step))
            F = F0 * (2 ** lamda)
            CR = 0.5 * (1 + random.random())

            if verbose:
                if cluster_id is not None:
                    print("[%d] step %d" % (cluster_id, step))
                else:
                    print("step %d" % step)

           # Perform local minimization of the best
            if self.local_minimization:
                self.swarm.minimize_best()

            self.swarm.cross_mutate(step,F,CR)
            self.swarm.selet(step,cluster_id)

            if save_intermediary:
                if (
                    save_all_intermediary
                    or (step % 10 == 0)
                    or step >= simulation_steps
                ):
                    self.swarm.save(step, saving_path)


    def report(self, output_file_name=""):
        """Writes to output_file_name if defined or to standard output the result of a DE execution."""
        output = "DE Execution Report:%s%s" % (os.linesep, os.linesep)
        output += "Seed: %s%s" % (self.random_number_generator.seed, os.linesep)
        output += "Number of Glowworms: %s%s" % (self.swarm.get_size(), os.linesep)
        if self.initial_coordinates_file:
            output += "Coordinates file: %s%s" % (
                self.initial_coordinates_file,
                os.linesep,
            )
        output += os.linesep
        output += "Algorithm parameters:%s" % os.linesep
        output += "Rho: %s%s" % (self.parameters.rho, os.linesep)
        output += "Gamma: %s%s" % (self.parameters.gamma, os.linesep)
        output += "Beta: %s%s" % (self.parameters.beta, os.linesep)
        output += "Initial Luciferin: %s%s" % (
            self.parameters.initial_luciferin,
            os.linesep,
        )
        output += "Initial Vision Range: %s%s" % (
            self.parameters.initial_vision_range,
            os.linesep,
        )
        output += "Maximum Vision Range: %s%s" % (
            self.parameters.max_vision_range,
            os.linesep,
        )
        output += "Maximum Number of Neighbors: %s%s" % (
            self.parameters.max_neighbors,
            os.linesep,
        )

        if output_file_name:
            output_file = open(output_file_name, "w")
            output_file.write(output)
            output_file.close()
        else:
            return output

    def __repr__(self):
        """String representation of the simulation"""
        return self.report()


class GSOBuilder(object):

    def __init__(self):
        self._initializer = None

    def create(
        self,
        number_of_glowworms,
        random_number_generator,
        gso_parameters,
        objective_function,
        bounding_box,
    ):
        self._initializer = RandomInitializer(
            [objective_function],
            number_of_glowworms,
            gso_parameters,
            bounding_box,
            random_number_generator,
        )
        return GSO(
            self._initializer.generate_glowworms(),
            gso_parameters,
            random_number_generator,
        )

    def create_from_file(
        self,
        number_of_glowworms,
        random_number_generator,
        gso_parameters,
        objective_function,
        bounding_box,
        initial_population_file,
    ):
        self._initializer = FromFileInitializer(
            [objective_function],
            number_of_glowworms,
            gso_parameters,
            bounding_box.dimension,
            initial_population_file,
        )
        return GSO(
            self._initializer.generate_glowworms(),
            gso_parameters,
            random_number_generator,
            initial_population_file,
        )



class LightdockGSOBuilder(object):
    """Creates a GSO simulation object for the LightDock framework"""

    def create_from_file(
        self,
        number_of_glowworms,
        random_number_generator,
        gso_parameters,
        adapters,
        scoring_functions,
        bounding_box,
        initial_population_file,
        step_translation,
        step_rotation,
        step_nmodes,
        local_minimization,
        anm_rec,
        anm_lig,
    ):
        """Creates a new GSO instance of the algorithm reading the initial position of the glowworms
        agents from initial_population_file and using the scoring function adapter.
        """
        self._initializer = LightdockFromFileInitializer(
            adapters,
            scoring_functions,
            number_of_glowworms,
            gso_parameters,
            bounding_box.dimension,
            initial_population_file,
            step_translation,
            step_rotation,
            random_number_generator,
            step_nmodes,
            anm_rec,
            anm_lig,
        )
        return GSO(
            self._initializer.generate_glowworms(),
            gso_parameters,
            random_number_generator,
            local_minimization=local_minimization,
        )
