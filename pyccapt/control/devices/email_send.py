"""Experiment-finished email notification.

Configuration lives in ``pyccapt/files/email_credentials.toml`` (TOML).
That file is gitignored. A checked-in template ``email_credentials.example.toml``
documents the schema; users copy it to the real name and fill in their SMTP
account.

Backwards compatibility: if the TOML file is absent but the legacy
``email_pass.txt`` exists, the password is read from there and the rest of
the SMTP config falls back to the historical Gmail defaults.

Failures (missing config, missing recipient, SMTP errors) raise rather
than silently swallow, so callers can log them via ``apt.log``.
"""

from __future__ import annotations

import datetime
import logging
import mimetypes
import smtplib
import ssl
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 path
    try:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    except ModuleNotFoundError:  # final fallback: no TOML available
        tomllib = None  # type: ignore[assignment]

from pyccapt.control.core import runtime

_LEGACY_PASSWORD_FILE = ("files", "email_pass.txt")
_CREDENTIALS_FILE = ("files", "email_credentials.toml")
_EXAMPLE_FILE = ("files", "email_credentials.example.toml")
_LOGO_CANDIDATES = (
    ("files", "logo2.png"),
    ("files", "logo.png"),
)
_DEFAULT_SMTP_SERVER = "smtp.gmail.com"
_DEFAULT_SMTP_PORT = 465
_LEGACY_SENDER = "oxcart.ap@gmail.com"

_log = logging.getLogger("pyccapt.email")


class EmailNotConfigured(RuntimeError):
    """Raised when no usable credentials file is present."""


def _project_path(*parts: str) -> Path:
    return runtime.project_path(*parts)


def _load_credentials() -> dict:
    """Return SMTP config as a dict. Raises EmailNotConfigured if no file exists.

    Order of resolution:

    1. ``pyccapt/files/email_credentials.toml`` (preferred).
    2. ``pyccapt/files/email_pass.txt`` (legacy; only the password lives there,
       the rest defaults to the historical Gmail config).
    """
    toml_path = _project_path(*_CREDENTIALS_FILE)
    if toml_path.is_file():
        if tomllib is None:
            raise EmailNotConfigured(
                f"Found {toml_path} but no TOML parser is available. Use Python 3.11+ or `pip install tomli`."
            )
        with open(toml_path, "rb") as fh:
            data = tomllib.load(fh)
        password = str(data.get("password", "")).strip()
        sender = str(data.get("sender_email", "")).strip()
        if not password or not sender:
            raise EmailNotConfigured(
                f"{toml_path} is missing 'sender_email' or 'password'. Compare with email_credentials.example.toml."
            )
        return {
            "smtp_server": str(data.get("smtp_server", _DEFAULT_SMTP_SERVER)).strip(),
            "smtp_port": int(data.get("smtp_port", _DEFAULT_SMTP_PORT)),
            "use_login": bool(data.get("use_login", True)),
            "sender_email": sender,
            "sender_display_name": str(data.get("sender_display_name", "")).strip(),
            "password": password,
            "cc": [str(addr).strip() for addr in data.get("cc", []) if str(addr).strip()],
            "attach_experiment_files": bool(data.get("attach_experiment_files", True)),
            "_source": str(toml_path),
        }

    legacy_path = _project_path(*_LEGACY_PASSWORD_FILE)
    if legacy_path.is_file():
        password = legacy_path.read_text(encoding="utf-8").strip().splitlines()
        password = password[0].strip() if password else ""
        if not password:
            raise EmailNotConfigured(f"{legacy_path} is empty.")
        _log.warning(
            "Using legacy %s. Migrate to %s — copy %s and fill it in.",
            legacy_path,
            toml_path,
            _project_path(*_EXAMPLE_FILE),
        )
        return {
            "smtp_server": _DEFAULT_SMTP_SERVER,
            "smtp_port": _DEFAULT_SMTP_PORT,
            "use_login": True,
            "sender_email": _LEGACY_SENDER,
            "sender_display_name": "",
            "password": password,
            "cc": [],
            "attach_experiment_files": True,
            "_source": str(legacy_path),
        }

    example = _project_path(*_EXAMPLE_FILE)
    raise EmailNotConfigured(
        "No email credentials found. Copy "
        f"{example} to {toml_path} and fill in your SMTP details, "
        "or place a legacy email_pass.txt next to it. The real config "
        "file is gitignored."
    )


def _resolve_logo() -> Path | None:
    for parts in _LOGO_CANDIDATES:
        candidate = _project_path(*parts)
        if candidate.is_file():
            return candidate
    return None


def _resolve_viz_image(variables) -> Path | None:
    """Newest Visualization full-window snapshot for this experiment.

    The Visualization process writes ``visualization_screenshot_<suffix>.png``
    into the experiment's ``meta_data`` folder — periodically (numeric
    suffix), at the end of the run (``_final``) and on demand for interim
    e-mails (``_email``). We attach the most recently written one so the
    e-mail carries a current picture of the detector / spectra. Returns
    None if no snapshot exists yet.
    """
    folder = _experiment_folder(variables)
    if folder is None:
        return None
    meta = folder / "meta_data"
    if not meta.is_dir():
        return None
    shots = list(meta.glob("visualization_screenshot_*.png"))
    if not shots:
        return None
    # Newest by modification time — the interim '_email' / final '_final'
    # shot is whatever was written last.
    return max(shots, key=lambda p: p.stat().st_mtime)


def _experiment_folder(variables) -> Path | None:
    path_value = getattr(variables, "path", None)
    if not path_value:
        return None
    folder = Path(str(path_value))
    return folder if folder.is_dir() else None


def _experiment_id(variables) -> str:
    """Best-effort experiment identifier for the attachment filename.

    Prefers ``exp_name`` (``<counter>_<date>_<electrode>_<name>``, set in
    experiment_state.prepare_experiment_output_paths), then the experiment
    folder name, then ``hdf5_data_name``.
    """
    for attr in ("exp_name", "hdf5_data_name"):
        value = str(getattr(variables, attr, "") or "").strip()
        if value:
            return value
    folder = _experiment_folder(variables)
    if folder is not None:
        return folder.name
    return "experiment"


def _attach_experiment_files(msg: MIMEMultipart, variables, interim: bool = False) -> list[str]:
    """Attach the Visualization snapshot (+ the details txt for final mails).

    The apt.log is never attached — the full run report is already in the
    e-mail body. For the final end-of-experiment e-mail (``interim=False``)
    the experiment-details text file (parameters.txt, delivered as
    ``experiment_details_<id>.txt``) is attached alongside the snapshot.
    Interim progress e-mails carry only the snapshot. Returns the names
    attached.
    """
    attached: list[str] = []
    folder = _experiment_folder(variables)
    if folder is None:
        return attached

    # On-disk name -> attachment (download) name.
    candidates = []
    if not interim:
        # Final report: include the experiment-details txt (not the log).
        details_name = f"experiment_details_{_experiment_id(variables)}.txt"
        candidates.append((folder / "parameters.txt", details_name))
    viz_path = _resolve_viz_image(variables)
    if viz_path is not None:
        candidates.append((viz_path, f"visualization_{_experiment_id(variables)}.png"))

    for path, attach_name in candidates:
        try:
            if not path.is_file():
                continue
            # Cap individual attachments at 10 MiB so a runaway log doesn't
            # block the SMTP server with a giant message.
            if path.stat().st_size > 10 * 1024 * 1024:
                _log.warning(
                    "Skipping email attachment %s (>10 MiB); see the experiment folder instead.",
                    path,
                )
                continue
            ctype, _ = mimetypes.guess_type(str(path))
            maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
            with open(path, "rb") as fh:
                part = MIMEApplication(fh.read(), _subtype=subtype)
            filename = attach_name or path.name
            part.add_header(
                "Content-Disposition",
                "attachment",
                filename=filename,
            )
            msg.attach(part)
            attached.append(filename)
        except Exception as exc:  # never let an attachment failure kill the email
            _log.warning("Could not attach %s: %s", path, exc)
    return attached


def _build_message(
    recipient: str,
    subject: str,
    body_text: str,
    config: dict,
    variables=None,
    interim: bool = False,
) -> tuple[MIMEMultipart, list[str], list[str]]:
    """Compose the MIME message. Returns (msg, all_recipients, attached_names)."""
    from_address = config["sender_email"]
    display_name = config.get("sender_display_name", "")
    cc_list: list[str] = list(config.get("cc", []))

    msg = MIMEMultipart("related")
    msg["From"] = formataddr((display_name, from_address)) if display_name else from_address
    msg["To"] = recipient
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject
    msg["Date"] = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S")

    # Inline-image content ID so the logo shows in the body, not just as
    # a separate attachment.
    logo_path = _resolve_logo()
    logo_html = ""
    if logo_path is not None:
        try:
            with open(logo_path, "rb") as fh:
                logo_data = fh.read()
            # Derive the MIME subtype from the filename extension. This is
            # more robust than letting MIMEImage sniff the byte content,
            # which raises if the bytes don't match a known image header.
            subtype = logo_path.suffix.lstrip(".").lower() or "png"
            if subtype == "jpg":
                subtype = "jpeg"
            logo_part = MIMEImage(logo_data, _subtype=subtype, name=logo_path.name)
            logo_part.add_header("Content-ID", "<pyccapt_logo>")
            logo_part.add_header("Content-Disposition", "inline", filename=logo_path.name)
            msg.attach(logo_part)
            logo_html = '<img src="cid:pyccapt_logo" alt="PyCCAPT" style="max-width:160px"/><br/>'
        except Exception as exc:
            _log.warning("Could not embed logo %s: %s", logo_path, exc)

    # Inline the latest Visualization snapshot (detector + spectra) so the
    # recipient sees the current state of the experiment in the body, and
    # keep the file as an attachment for full resolution.
    viz_html = ""
    viz_path = _resolve_viz_image(variables) if variables is not None else None
    if viz_path is not None:
        try:
            with open(viz_path, "rb") as fh:
                viz_data = fh.read()
            viz_part = MIMEImage(viz_data, _subtype="png", name=viz_path.name)
            viz_part.add_header("Content-ID", "<pyccapt_viz>")
            viz_part.add_header("Content-Disposition", "inline", filename=viz_path.name)
            msg.attach(viz_part)
            viz_html = (
                '<p style="margin:8px 0 4px 0;font-weight:bold">Visualization snapshot</p>'
                '<img src="cid:pyccapt_viz" alt="Visualization snapshot" '
                'style="max-width:640px;border:1px solid #ccc"/><br/>'
            )
        except Exception as exc:
            _log.warning("Could not embed visualization image %s: %s", viz_path, exc)

    # Multipart/alternative inside the related container so clients pick the
    # best representation. Plain text is the source of truth; HTML is just a
    # nicer rendering.
    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(body_text, "plain", _charset="utf-8"))
    html_body = (
        "<html><body style='font-family:Segoe UI,Arial,sans-serif'>"
        f"{logo_html}"
        "<pre style='font-family:Consolas,Menlo,monospace;font-size:12px'>"
        f"{body_text}"
        "</pre>"
        f"{viz_html}"
        "</body></html>"
    )
    alternative.attach(MIMEText(html_body, "html", _charset="utf-8"))
    msg.attach(alternative)

    attached_names: list[str] = []
    if config.get("attach_experiment_files", True) and variables is not None:
        attached_names = _attach_experiment_files(msg, variables, interim=interim)

    all_recipients = [recipient] + cc_list
    return msg, all_recipients, attached_names


def send_email(email: str, subject: str, message: str, variables=None, interim: bool = False) -> list[str]:
    """Send the experiment notification email.

    Args:
            email: Recipient address (the operator's address from the GUI).
            subject: Email subject.
            message: Plain-text body.
            variables: Optional shared experiment variables; if provided, the
                    latest Visualization snapshot is inlined/attached.
            interim: True for a mid-run progress e-mail (snapshot only); False
                    for the final report (snapshot + experiment-details txt).

    Returns:
            List of attached filenames (may be empty).

    Raises:
            EmailNotConfigured: if no credentials file is set up.
            ValueError: if the recipient address is empty.
            smtplib.SMTPException / OSError: on SMTP/network failure.
    """
    recipient = (email or "").strip()
    if not recipient or "@" not in recipient:
        raise ValueError(f"No valid recipient address: {email!r}")

    config = _load_credentials()
    msg, all_recipients, attached = _build_message(
        recipient, subject, message, config, variables=variables, interim=interim
    )

    port = int(config["smtp_port"])
    server_name = config["smtp_server"]
    use_login = config.get("use_login", True)
    context = ssl.create_default_context()

    # 465 = implicit SSL; everything else = STARTTLS upgrade.
    if port == 465:
        smtp_cls = lambda: smtplib.SMTP_SSL(server_name, port, context=context, timeout=30)
    else:

        def smtp_cls():
            s = smtplib.SMTP(server_name, port, timeout=30)
            s.ehlo()
            try:
                s.starttls(context=context)
                s.ehlo()
            except smtplib.SMTPException:
                # Server may be plain SMTP without TLS — still OK in trusted lab nets.
                _log.warning("STARTTLS not available on %s:%s", server_name, port)
            return s

    with smtp_cls() as server:
        if use_login:
            server.login(config["sender_email"], config["password"])
        server.sendmail(config["sender_email"], all_recipients, msg.as_string())

    _log.info(
        "Sent notification to %s (cc: %s) via %s; attached: %s",
        recipient,
        ", ".join(config.get("cc") or []) or "none",
        server_name,
        ", ".join(attached) or "none",
    )
    return attached
