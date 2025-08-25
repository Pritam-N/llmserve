from __future__ import annotations
import asyncio
import os, sys, subprocess
import typer
from rich import print as rprint
from . import __version__
from .config import load_manifest
from .runner import Orchestrator
from .deploy.k8sgen import render_manifests, write_out

app = typer.Typer(add_completion=False, help="LLMServe CLI")

@app.callback()
def _global(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    if verbose:
        os.environ["LOGLEVEL"] = "DEBUG"

@app.command(help="Run locally OR render/apply K8s manifests.")
def up(
    manifest: str = typer.Option("llmserve.yaml", "--manifest", "-f", help="Path to manifest YAML"),
    host: str = typer.Option("0.0.0.0", help="Bind host (local mode)"),
    port: int = typer.Option(8000, help="HTTP port (local mode)"),
    metrics_port: int = typer.Option(9400, help="Prometheus port (local mode)"),
    mode: str = typer.Option("local", "--mode", help="local|k8s"),
    namespace: str = typer.Option("llmserve", "--namespace"),
    image: str = typer.Option(None, "--image", help="Container image (k8s mode)"),
    svc_type: str = typer.Option("LoadBalancer", "--service-type", help="ClusterIP|NodePort|LoadBalancer"),
    out_dir: str = typer.Option("deploy_out/k8s", "--out-dir", help="Where to write generated YAML in k8s mode"),
    apply: bool = typer.Option(False, "--apply", help="Run `kubectl apply -f` on the out-dir"),
):
    spec = load_manifest(manifest)

    if mode == "local":
        orch = Orchestrator(spec)
        try:
            asyncio.run(orch.run_local(host=host, port=port, metrics_port=metrics_port))
        except KeyboardInterrupt:
            rprint("[yellow]Shutting down...[/yellow]")
        return

    if mode != "k8s":
        rprint(f"[red]Unknown mode {mode}[/red]"); raise typer.Exit(1)

    # Render manifests
    docs = render_manifests(spec, namespace=namespace, image=image, replicas=None)
    with open(manifest, "r", encoding="utf-8") as f:
        manitext = f.read()
    outdir = write_out(out_dir, docs, manitext)
    rprint(f"[green]Rendered K8s manifests to[/green] {outdir}")

    if apply:
        cmd = ["kubectl", "apply", "-f", str(outdir)]
        rprint(f"[cyan]Applying:[/cyan] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            rprint(f"[red]kubectl apply failed:[/red] {e}")
            raise typer.Exit(1)

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