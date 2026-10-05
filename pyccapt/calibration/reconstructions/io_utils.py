"""Shared I/O helpers for reconstruction plotting outputs."""

from __future__ import annotations

import json
from pathlib import Path

import plotly
import plotly.io as pio

from pyccapt.calibration.path_utils import build_output_path, save_figure


_CAMERA_GIF_EXPORTER = r"""
<section id="pyccapt-camera-gif" style="font: 14px sans-serif; margin: 10px 0; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
  <strong>Current-camera GIF</strong>
  <label>Frames <input id="pyccapt-gif-frames" type="number" min="4" max="120" value="20" style="width: 58px"></label>
  <label>FPS <input id="pyccapt-gif-fps" type="number" min="1" max="30" value="2" style="width: 48px"></label>
  <button id="pyccapt-gif-save">Save rotating GIF</button>
  <button id="pyccapt-gif-cancel" disabled>Cancel GIF</button>
  <span id="pyccapt-gif-status" aria-live="polite"></span>
</section>
<script>
(() => {
  const baseName = __PYCCAPT_GIF_BASENAME__;
  const workerUrl = 'https://cdn.jsdelivr.net/npm/gif.js.optimized/dist/gif.worker.js';
  const graph = document.querySelector('.plotly-graph-div');
  const framesInput = document.getElementById('pyccapt-gif-frames');
  const fpsInput = document.getElementById('pyccapt-gif-fps');
  const saveButton = document.getElementById('pyccapt-gif-save');
  const cancelButton = document.getElementById('pyccapt-gif-cancel');
  const status = document.getElementById('pyccapt-gif-status');
  let activeGif = null;
  let cancelled = false;

  const setStatus = (text) => { status.textContent = text; };
  const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
  const withTimeout = async (promise, milliseconds, task) => {
    let timer;
    try {
      return await Promise.race([
        promise,
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${task} timed out`)), milliseconds); }),
      ]);
    } finally { clearTimeout(timer); }
  };
  const imageFromDataUrl = (url) => new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Could not read a rendered GIF frame'));
    image.src = url;
  });
  const loadGifEncoder = () => {
    if (window.GIF) return Promise.resolve(window.GIF);
    return withTimeout(new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/gif.js.optimized/dist/gif.js';
      script.onload = () => window.GIF ? resolve(window.GIF) : reject(new Error('GIF encoder did not load'));
      script.onerror = () => reject(new Error('GIF encoder could not be downloaded; check internet access'));
      document.head.appendChild(script);
    }), 20000, 'GIF encoder download');
  };
  const createWorkerScriptUrl = async () => {
    const response = await withTimeout(fetch(workerUrl), 20000, 'GIF worker download');
    if (!response.ok) throw new Error(`GIF worker could not be downloaded (${response.status})`);
    const source = await response.text();
    return URL.createObjectURL(new Blob([source], {type: 'text/javascript'}));
  };
  const download = (blob) => {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${baseName}_current_camera.gif`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  };
  const restore = async (camera, showLegend) => {
    await Plotly.relayout(graph, {'scene.camera': camera, showlegend: showLegend});
  };

  cancelButton.addEventListener('click', () => {
    cancelled = true;
    if (activeGif && activeGif.abort) activeGif.abort();
    setStatus('Cancelling GIF…');
  });
  saveButton.addEventListener('click', async () => {
    if (!graph) { setStatus('No Plotly 3D figure was found in this HTML file.'); return; }
    const frameCount = Math.max(4, Math.min(120, Number.parseInt(framesInput.value, 10) || 20));
    const fps = Math.max(1, Math.min(30, Number.parseInt(fpsInput.value, 10) || 2));
    const scene = graph._fullLayout && graph._fullLayout.scene;
    const originalCamera = JSON.parse(JSON.stringify((scene && scene.camera) || {eye: {x: 1.25, y: 1.25, z: 1.25}}));
    const originalLegend = graph.layout.showlegend !== false;
    const eye = originalCamera.eye || {x: 1.25, y: 1.25, z: 1.25};
    const width = Math.max(320, graph.clientWidth || 700);
    const height = Math.max(240, graph.clientHeight || 500);
    saveButton.disabled = true;
    cancelButton.disabled = false;
    cancelled = false;
    let workerScriptUrl = null;
    try {
      setStatus('Preparing GIF encoder…');
      const [GIF, localWorkerScriptUrl] = await Promise.all([loadGifEncoder(), createWorkerScriptUrl()]);
      workerScriptUrl = localWorkerScriptUrl;
      const gif = activeGif = new GIF({workers: 2, quality: 10, width, height, workerScript: workerScriptUrl});
      await Plotly.relayout(graph, {showlegend: false});
      for (let frame = 0; frame < frameCount; frame += 1) {
        if (cancelled) throw new Error('GIF export cancelled');
        const theta = 2 * Math.PI * frame / frameCount;
        const camera = JSON.parse(JSON.stringify(originalCamera));
        camera.eye = {x: eye.x * Math.cos(theta) - eye.y * Math.sin(theta), y: eye.x * Math.sin(theta) + eye.y * Math.cos(theta), z: eye.z};
        await withTimeout(Plotly.relayout(graph, {'scene.camera': camera}), 10000, 'Camera update');
        await sleep(20);
        const dataUrl = await withTimeout(Plotly.toImage(graph, {format: 'png', width, height, scale: 1}), 30000, 'Frame render');
        gif.addFrame(await imageFromDataUrl(dataUrl), {delay: Math.round(1000 / fps), copy: true});
        setStatus(`Rendering frame ${frame + 1}/${frameCount}…`);
      }
      await restore(originalCamera, originalLegend);
      setStatus('Encoding GIF…');
      const result = await withTimeout(new Promise((resolve, reject) => {
        gif.on('finished', resolve);
        gif.on('abort', () => reject(new Error('GIF export cancelled')));
        gif.render();
      }), 120000, 'GIF encoding');
      download(result);
      setStatus(`Saved ${frameCount}-frame GIF at ${fps} FPS.`);
    } catch (error) {
      await restore(originalCamera, originalLegend).catch(() => {});
      setStatus(`GIF was not saved: ${error.message}`);
    } finally {
      if (workerScriptUrl) URL.revokeObjectURL(workerScriptUrl);
      activeGif = null;
      saveButton.disabled = false;
      cancelButton.disabled = true;
    }
  });
})();
</script>
"""


def _add_camera_gif_exporter(path: str | Path, filename: str) -> None:
    """Append a browser-side current-camera GIF control to a Plotly HTML file."""
    output_path = Path(path)
    html = output_path.read_text(encoding="utf-8")
    exporter = _CAMERA_GIF_EXPORTER.replace("__PYCCAPT_GIF_BASENAME__", json.dumps(Path(filename).stem))
    marker = "</body>"
    output_path.write_text(
        html.replace(marker, f"{exporter}\n{marker}") if marker in html else html + exporter,
        encoding="utf-8",
    )


def resolve_result_file(variables, filename: str) -> str:
    """Resolve an output filename against the reconstruction result directory."""
    resolver = getattr(variables, "resolve_result_file", None)
    if callable(resolver):
        return resolver(filename)

    result_path = str(getattr(variables, "result_path", "")).strip()
    if result_path:
        return str(build_output_path(result_path, filename))
    return str(Path(filename))


def save_matplotlib_figure(
    fig,
    variables,
    *,
    stem: str,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 600,
    **savefig_kwargs,
) -> list[Path]:
    """Save a matplotlib figure to the reconstruction result directory."""
    output_dir = getattr(variables, "result_path", "") or "."
    return save_figure(fig, directory=output_dir, stem=stem, formats=formats, dpi=dpi, **savefig_kwargs)


def write_plotly_html(
    fig,
    variables,
    filename: str,
    *,
    include_mathjax: str = "cdn",
    add_camera_gif_exporter: bool = False,
) -> None:
    """Write a Plotly figure to HTML inside the reconstruction result directory."""
    output_path = resolve_result_file(variables, filename)
    pio.write_html(fig, output_path, include_mathjax=include_mathjax)
    if add_camera_gif_exporter:
        _add_camera_gif_exporter(output_path, filename)


def write_plotly_image(
    fig,
    variables,
    filename: str,
    *,
    scale: int = 3,
    image_format: str | None = None,
) -> None:
    """Write a Plotly static image inside the reconstruction result directory."""
    resolved = resolve_result_file(variables, filename)
    if image_format is None:
        image_format = Path(filename).suffix.lstrip(".")
    pio.write_image(fig, resolved, scale=scale, format=image_format)


def save_gif(images, variables, filename: str, *, fps: int = 2) -> None:
    """Save a sequence of frames to GIF inside the reconstruction result directory."""
    from PIL import Image

    if not images:
        raise ValueError("Cannot save a GIF without frames")
    if fps <= 0:
        raise ValueError("GIF frames per second must be greater than zero")
    path = resolve_result_file(variables, filename)
    duration_ms = int(1000 / fps)
    frames = [Image.fromarray(img) if not isinstance(img, Image.Image) else img for img in images]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        # Palette optimization is disproportionately slow for large 3D PNG
        # frames and makes the UI appear frozen after rendering completes.
        optimize=False,
        loop=0,
        duration=duration_ms,
    )


def save_plotly_animation(
    fig,
    variables,
    *,
    filename: str,
    show_link: bool = True,
    auto_open: bool = False,
    include_mathjax: str = "cdn",
    add_camera_gif_exporter: bool = False,
) -> None:
    """Save an interactive Plotly animation HTML file to the result directory."""
    output_path = resolve_result_file(variables, filename)
    plotly.offline.plot(
        fig,
        filename=output_path,
        show_link=show_link,
        auto_open=auto_open,
        include_mathjax=include_mathjax,
    )
    if add_camera_gif_exporter:
        _add_camera_gif_exporter(output_path, filename)
