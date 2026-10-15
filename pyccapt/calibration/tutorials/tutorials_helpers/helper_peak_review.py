"""Peak-review queue UI. Detection/ranking proposes; buttons record user decisions."""
from __future__ import annotations

from copy import copy, deepcopy
from html import escape
import json

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from pyccapt.calibration.core.isotope_catalogue import get_isotope_catalogue
from pyccapt.calibration.core.exceptions import CalibrationStateError
from pyccapt.calibration.tutorials.tutorials_helpers.guided_jobs import NotebookJob


def show_peak_review(workflow, *, display_panel=True, on_change=lambda: None):
    workflow.begin_review()
    output, spectrum = widgets.Output(), widgets.Output()
    status, warnings = widgets.HTML(), widgets.HTML()
    elements = widgets.Text(value="Fe,Ni,Cr", description="Elements:")
    charge = widgets.BoundedIntText(value=3, min=1, max=8, description="Max charge:")
    atoms = widgets.BoundedIntText(value=2, min=1, max=5, description="Max atoms:")
    bin_width = widgets.BoundedFloatText(value=.02, min=.00001, max=10, description="Bin (Da):")
    prominence = widgets.BoundedFloatText(value=20, min=1, max=1e9, description="Prominence:")
    tolerance = widgets.BoundedFloatText(value=.15, min=.00001, max=5, description="Tolerance Da:")
    minimum = widgets.FloatText(value=0, description="Mass min:")
    maximum = widgets.FloatText(value=max(1., float(workflow.review.sorted_mass[-1])+.02), description="Mass max:")
    peaks = widgets.Select(options=[], description="Queue:", rows=10, layout=widgets.Layout(width="380px"))
    candidate = widgets.Dropdown(options=[], description="Identity:", layout=widgets.Layout(width="690px"))
    lower, upper = widgets.FloatText(description="Lower Da:"), widgets.FloatText(description="Upper Da:")
    auto_zoom = widgets.Checkbox(value=True, description="Auto zoom to selected peak")
    view_lower, view_upper = widgets.FloatText(description="View min Da:"), widgets.FloatText(value=1., description="View max Da:")
    show_view = widgets.Button(description="Apply view")
    split_at = widgets.FloatText(description="Split at Da:")
    reviewed = widgets.Checkbox(value=False, description="I reviewed identity ambiguity / overlapping bounds", indent=False)
    family_ids = widgets.SelectMultiple(options=[], description="Family/merge:", rows=5, layout=widgets.Layout(width="600px"))
    family_checked = widgets.Checkbox(value=False, description="I reviewed isotope-family identity and spacing consistency", indent=False)
    reference = widgets.Dropdown(options=[], description="Template:", layout=widgets.Layout(width="600px"))
    progress = widgets.FloatProgress(min=0, max=1, description="Progress:")
    progress_text = widgets.Label()
    buttons = {name: widgets.Button(description=name) for name in
        ("Propose peaks", "Choose identity", "Accept identity", "Reject identity", "Not a peak", "Defer", "Next",
         "Edit bounds", "Split", "Merge selected", "Undo review", "Accept family", "Profile evidence", "Mixture evidence",
         "Preview ranges", "Accept ranges", "Cancel")}
    job = NotebookJob()
    refreshing = False
    if workflow.review.catalogue_spec is not None:
        selected_elements, selected_charges, selected_atoms, _ = workflow.review.catalogue_spec
        elements.value, charge.value, atoms.value = ",".join(selected_elements), max(selected_charges), selected_atoms
    tolerance.value = workflow.review.tolerance

    def message(value):
        with output:
            output.clear_output(wait=True)
            print(json.dumps(value, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)))

    def refresh():
        nonlocal refreshing
        refreshing = True
        queue = workflow.review
        if queue is None:
            status.value = "Calibration changed: reopen the peak-review queue for the accepted spectrum."
            for button in buttons.values():
                button.disabled = True
            refreshing = False
            return
        current_id = peaks.value
        options = [(f"{r['center']:.4f} Da | {r['status']} | {r['chosen']['ion'] if r['chosen'] else 'unconfirmed'}", r["id"])
                   for r in queue.records]
        peaks.options = options
        if current_id in [r["id"] for r in queue.records]:
            peaks.value = current_id
        elif options:
            peaks.value = options[min(queue.current, len(options)-1)][1]
        family_ids.options = options
        reference.options = [(r["chosen"]["ion"]+f" ({r['center']:.4f})", r["id"])
                             for r in queue.records if r["status"] == "accepted"]
        status.value = (f"<b>{len(queue.records)} peaks</b> · {sum(r['status']=='accepted' for r in queue.records)} accepted · "
                        f"{sum(not r['locked'] for r in queue.records)} awaiting review. "
                        "Candidate order is mass proximity and simplicity, <b>not probability</b>.")
        for name, button in buttons.items():
            button.disabled = job.running or (not options and name not in {"Propose peaks", "Cancel"})
        buttons["Cancel"].disabled = not job.running
        refreshing = False
        show_current()
        on_change()

    def show_current(*_):
        if refreshing or not peaks.value:
            return
        queue = workflow.review
        row = queue._get(peaks.value)
        queue.current = next(i for i, r in enumerate(queue.records) if r["id"] == row["id"])
        candidate.options = [(f"{item['ion']} | Δm={row['center']-item['mass']:+.5f} Da | {sum(item['complex'])} atom(s)", item["key"])
                             for item in row["candidates"] if item["key"] not in row["rejected"]]
        if row["chosen"] and row["chosen"]["key"] in [value for _, value in candidate.options]:
            candidate.value = row["chosen"]["key"]
        else:
            candidate.value = candidate.options[0][1] if candidate.options else None
        lower.value, upper.value, split_at.value = row["lower"], row["upper"], row["center"]
        reviewed.value = bool(row["reviewed_ambiguity"])
        items = queue.warnings(row["id"])
        warnings.value = "<br>".join(escape(item) for item in items) or "No overlapping window detected. Identity still requires review."
        with spectrum:
            spectrum.clear_output(wait=True)
            width = max(row["upper"]-row["lower"], .1)
            if auto_zoom.value:
                view_lower.value, view_upper.value = max(0, row["lower"]-width), row["upper"]+width
            lo, hi = view_lower.value, view_upper.value
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                print("View bounds must be finite and increasing")
                return
            counts, edges = queue.histogram(max(width/150, .0001), lo, hi)
            fig, axis = plt.subplots(figsize=(10, 3))
            axis.step((edges[:-1]+edges[1:])/2, counts, where="mid", color="black")
            axis.axvspan(row["lower"], row["upper"], color="tab:blue", alpha=.15)
            for option in row["candidates"][:8]:
                axis.axvline(option["mass"], alpha=.35, linestyle="--")
            axis.set(xlabel="Mass-to-charge (Da)", ylabel="Events / bin", title=f"{row['center']:.4f} Da · {queue.count(row['id']):,} events in window")
            display(fig)
            plt.close(fig)
            curve = row["evidence"].get("mixture_curve")
            if curve:
                fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
                axes[0].step(curve["mass"], curve["observed"], where="mid", label="Observed")
                axes[0].plot(curve["mass"], curve["prediction"], label="Conditional family fit")
                axes[0].legend()
                axes[1].plot(curve["mass"], curve["residual"], label="Observed − fit")
                axes[1].axhline(0, color="gray")
                axes[1].set(xlabel="Mass-to-charge (Da)", ylabel="Residual counts")
                display(fig)
                plt.close(fig)
        message({"decision": row["status"], "chosen_identity": row["chosen"],
                 "evidence": {k: v for k, v in row["evidence"].items() if k != "mixture_curve"}})

    def act(operation):
        def callback(_):
            try:
                value = operation()
                family_checked.value = False
                refresh()
                if value is not None:
                    message(value)
            except Exception as error:
                message(f"{type(error).__name__}: {error}")
        return callback

    def launch(operation, consume):
        from pyccapt.calibration.core.guided_workflow import _pack
        from pyccapt.calibration.data_tools.analysis_session import canonical_json
        revision = workflow.revision
        queue = workflow.review
        decision_state = canonical_json(_pack(queue.to_state(), {}))
        def receive(result):
            workflow._check()
            workflow._revision_check(revision)
            if workflow.review is not queue or canonical_json(_pack(queue.to_state(), {})) != decision_state:
                raise CalibrationStateError("Peak decisions changed during this operation; result discarded, rerun proposals/evidence")
            consume(result)
        job.start(operation, on_result=receive, on_error=lambda error: message(f"{type(error).__name__}: {error}"),
                  on_progress=lambda fraction, text: (setattr(progress, "value", fraction), setattr(progress_text, "value", text)),
                  on_finished=refresh)
        refresh()

    def fork_queue():
        result = copy(workflow.review)
        result.records, result.history = deepcopy(result.records), deepcopy(result.history)
        result._histograms = result._histograms.copy()
        result._catalogues = result._catalogues.copy()
        return result

    def propose():
        config = dict(elements=elements.value, charges=tuple(range(1, charge.value+1)), max_atoms=atoms.value)
        settings = dict(bin_width=bin_width.value, prominence=prominence.value, tolerance=tolerance.value,
                        lower=minimum.value, upper=maximum.value)
        working = fork_queue()
        def operation(cancel, report):
            catalogue = get_isotope_catalogue(**config, cancel=cancel, progress=report)
            working.propose(catalogue, **settings, cancel=cancel, progress=report)
            return working
        launch(operation, lambda result: setattr(workflow, "review", result))

    def accept_identity():
        queue = workflow.review
        row = queue._get(peaks.value)
        if row["chosen"] is None or row["chosen"]["key"] != candidate.value:
            queue.choose(peaks.value, candidate.value)
        queue.decide(peaks.value, "accept", ambiguity_reviewed=reviewed.value)
        row = queue.next()
        if row:
            peaks.value = row["id"]

    def next_peak():
        row = workflow.review.next()
        if row:
            peaks.value = row["id"]

    def evidence(mixture=False):
        working, selected = fork_queue(), peaks.value
        ids, template, confirmed = list(family_ids.value), reference.value, family_checked.value
        def operation(cancel, report):
            report(0., "Fitting optional evidence; cancellation discards the result after this solver returns")
            result = working.mixture_evidence(ids, template, identities_reviewed=confirmed) if mixture else working.profile_evidence(selected)
            report(1., "Evidence ready; no decisions accepted")
            return working, result.summary() if mixture else result
        def consume(result):
            workflow.review = result[0]
            message(result[1])
        launch(operation, consume)

    actions = {"Propose peaks": propose, "Choose identity": lambda: workflow.review.choose(peaks.value, candidate.value),
        "Accept identity": accept_identity, "Reject identity": lambda: workflow.review.decide(peaks.value, "reject_identity"),
        "Not a peak": lambda: workflow.review.decide(peaks.value, "not_peak"), "Defer": lambda: workflow.review.decide(peaks.value, "defer"),
        "Next": next_peak, "Edit bounds": lambda: workflow.review.edit_bounds(peaks.value, lower.value, upper.value),
        "Split": lambda: workflow.review.split(peaks.value, split_at.value), "Merge selected": lambda: workflow.review.merge(list(family_ids.value)),
        "Undo review": lambda: workflow.review.undo(), "Accept family": lambda: workflow.review.accept_family(list(family_ids.value),
            consistency_reviewed=family_checked.value, ambiguity_reviewed=reviewed.value),
        "Profile evidence": evidence, "Mixture evidence": lambda: evidence(True),
        "Preview ranges": workflow.preview_ranges, "Accept ranges": lambda: workflow.accept("ranges"), "Cancel": job.cancel}
    for name, operation in actions.items():
        buttons[name].on_click(act(operation))
    peaks.observe(show_current, names="value")
    family_ids.observe(lambda _: setattr(family_checked, "value", False), names="value")
    reference.observe(lambda _: setattr(family_checked, "value", False), names="value")
    candidate.observe(lambda _: (setattr(reviewed, "value", False), setattr(family_checked, "value", False)), names="value")
    show_view.on_click(lambda _: show_current())
    refresh()
    panel = widgets.VBox([status, widgets.HBox([elements, charge, atoms]), widgets.HBox([bin_width, prominence, tolerance]),
        widgets.HBox([minimum, maximum, buttons["Propose peaks"], buttons["Cancel"]]), progress, progress_text,
        widgets.HBox([peaks, widgets.VBox([candidate, warnings, reviewed])]),
        widgets.HBox([auto_zoom, view_lower, view_upper, show_view]), spectrum,
        widgets.HBox([buttons[name] for name in ("Choose identity", "Accept identity", "Reject identity", "Not a peak", "Defer", "Next", "Undo review")]),
        widgets.HBox([lower, upper, buttons["Edit bounds"], split_at, buttons["Split"]]),
        family_ids, family_checked, reference, widgets.HBox([buttons[name] for name in ("Merge selected", "Accept family", "Profile evidence", "Mixture evidence")]),
        widgets.HBox([buttons["Preview ranges"], buttons["Accept ranges"]]), output])
    panel.controls = dict(buttons, peaks=peaks, candidate=candidate, elements=elements, job=job, refresh=refresh,
                          reviewed=reviewed, family_ids=family_ids, family_checked=family_checked,
                          charge=charge, atoms=atoms, bin_width=bin_width, prominence=prominence,
                          tolerance=tolerance, minimum=minimum, maximum=maximum, lower=lower, upper=upper)
    if display_panel:
        display(panel)
    return panel
