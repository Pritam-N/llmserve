from __future__ import annotations
import asyncio
import os
import sys
import typer
from rich import print as rprint
from . import __version__
from .config import load_manifest
from .runner import Orchestrator

app = typer.Typer(add_completion=False, help="LLMServe CLI")

@app.callback()
def _global(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    if verbose:
        os.environ["LOGLEVEL"] = "DEBUG"

@app.command(help="Run everything locally in one process (API + stubs).")
def up(
    manifest: str = typer.Option("llmserve.yaml", "--manifest", "-f", help="Path to manifest YAML"),
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="HTTP port"),
    metrics_port: int = typer.Option(9400, help="Prometheus port"),
):
    spec = load_manifest(manifest)
    orch = Orchestrator(spec)
    try:
        asyncio.run(orch.run_local(host=host, port=port, metrics_port=metrics_port))
    except KeyboardInterrupt:
        rprint("[yellow]Shutting down...[/yellow]")

@app.command(help="Validate manifest and print a deploy plan (stub).")
def apply(
    manifest: str = typer.Option("llmserve.yaml", "--manifest", "-f"),
):
    spec = load_manifest(manifest)
    rprint("[green]Manifest OK[/green]")
    rprint("Suggested deploy: deploy/k8s/* (use ConfigMap to mount your manifest)")

@app.command(help="Quick health/status (stub).")
def status():
    rprint("router: OK | prefill: 0/0 (stub) | decode: 0/0 (stub) | kv-manager: 0/0 (stub)")

@app.command(help="Show CLI/package version.")
def version():
    rprint(f"llmserve {__version__}")

if __name__ == "__main__":
    app()