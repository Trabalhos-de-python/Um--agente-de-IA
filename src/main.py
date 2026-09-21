from runpy import run_module

if __name__ == "__main__":
    run_module("um_agente_de_ia.main", run_name="__main__")
else:
    from um_agente_de_ia.main import *
