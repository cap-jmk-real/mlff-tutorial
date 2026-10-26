import subprocess


def run_xtb(
    substrate_file: str,
    adstrate_file: str,
    num_proccesses: int = 4,
    xtb_path: str = "xtb",
):

    try:
        out = subprocess.run(
            [f"{xtb_path} dock -P {num_proccesses} {substrate_file} {adstrate_file}"],
            capture_output=True,
            shell=True,
        )
        xtb = out.stdout.decode()
    except Exception as e:
        raise Warning("XTB run did not terminate")
    return xtb
