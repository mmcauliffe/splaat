from rich.traceback import install

from splaat.command_line import splaat_cli

install(show_locals=True)
splaat_cli()
