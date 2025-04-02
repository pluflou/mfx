from typing import Literal
from pydantic import validate_call
from bluesky import RunEngine


Diagnostics = Literal["XCS1", "DG1", "DG2"]
Methods = Literal["Xopt", "blop"]
Devices = Literal["yag", "wave8"]
Turbo = Literal["safety", "optimize"]


class Align:
    @validate_call
    def __init__(self, mirror_pitch: list[float] = [-549.0, -546.0]):
        self.mirror_pitch: list[float] = mirror_pitch

    @validate_call
    def beam(
            self,
            with_goal: float,
            on_diagnostic: Diagnostics = "DG1",
            with_method: Methods = "Xopt",
            using_device: Devices = "yag",
            xopt_turbo_option: Turbo = "safety",
            xopt_rand_evaluate: int = 3,
            xopt_steps: int = 10,
            blop_qr_n: int = 16,
            blop_qei_n: int = 16,
            blop_qei_iterations: int = 5
            ):
        """Perform Beam Alignment

        Parameters:

            with_goal (float): goal to align to.
            on_diagnostic (str): diagnostic to use for alignment. Options: "XCS1, DG1, DG2".
            with_method (str): method to use for alignment. Options: "blop, Xopt".
            using_device (str): device to use for alignment. Options: "yag, wave8".
            xopt_turbo_option (str): xopt turbo controller option. Options: "safety, optimize".
            xopt_rand_evaluate (int): number of random evaluations to perform.
            xopt_steps (int): number of steps to perform.
            blop_qr_n (int): number of qr iterations to perform.
            blop_qei_n (int): number of qei iterations to perform.
            blop_qei_iterations (int): number of iterations to perform.
        """
        if with_method == 'Xopt':
            from .xopt_scans import get_xopt_obj, init_devices
            xopt = get_xopt_obj(
                device_type=using_device,
                location=on_diagnostic,
                goal=with_goal,
                xopt_generator_turbo_controller=xopt_turbo_option,
            )
            customized_boundaries = {'mirror_pitch': self.mirror_pitch}
            xopt.random_evaluate(xopt_rand_evaluate, custom_bounds=customized_boundaries)
            print(xopt.data)
            for num in range(xopt_steps):
                print(f"Step {num + 1}")
                xopt.step()
                print(xopt.data)
            _, val, params = xopt.vocs.select_best(xopt.data)
            print(f"Best objective value {val}")
            print(f"Best point {params}")
            mirror_pitch = init_devices()["mr1l4_homs"].pitch
            mirror_pitch.set(params["mirror_pitch"]).wait(timeout=20)
            print(f"pitch is at {mirror_pitch.position}")
            xopt.data.plot(y=xopt.vocs.objective_names)
            return xopt
        elif with_method == 'blop':
            from .blop_scans import get_blop_agent
            try:
                from mfx.db import RE
            except ImportError:
                RE = RunEngine({})
            agent = get_blop_agent(on_diagnostic.lower(), wave8_xpos=with_goal)
            RE(agent.learn("qr", n=blop_qr_n))
            RE(agent.learn("qei", n=blop_qei_n, iterations=blop_qei_iterations))
            RE(agent.go_to_best())
            agent.plot_objectives()
            return agent
