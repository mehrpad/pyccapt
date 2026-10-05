"""Durable pre-experiment transfer journal, copied into each started dataset."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from pyccapt.control.core import runtime


def begin_transfer_journal(variables, queue_index, settings, current_m, target_m):
    directory = runtime.project_path('data', 'alignment_sequences', variables.alignment_sequence_id,
                                     f'{queue_index + 1:03d}_sample_{variables.alignment_sample}')
    directory.mkdir(parents=True, exist_ok=False)
    variables.alignment_transfer_path = str(directory / 'alignment_transfer.jsonl')
    transfer_event(variables, 'transfer_start', settings=settings,
                   current_m=current_m, saved_position_m=target_m, queue_index=queue_index + 1)


def transfer_event(variables, event, **details):
    path = getattr(variables, 'alignment_transfer_path', '')
    if not path:
        return
    record = {'event': event, 'utc': datetime.now(timezone.utc).isoformat(),
              'monotonic_s': time.monotonic(), 'sequence_id': variables.alignment_sequence_id,
              'sample': variables.alignment_sample, **details}
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


def copy_transfer_journal(variables, meta_path):
    source = getattr(variables, 'alignment_transfer_path', '')
    if getattr(variables, 'automatic_alignment_enabled', False) and source:
        destination = Path(meta_path) / 'alignment_transfer.jsonl'
        temporary = destination.with_suffix('.jsonl.tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
