from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from pyccapt.calibration.data_tools import data_tools
from pyccapt.calibration.mc import tof_tools


def _loader_progress(*, total: int, enabled: bool, desc: str):
	return tqdm(
		total=total,
		desc=desc,
		unit="stage",
		dynamic_ncols=True,
		disable=not enabled,
		leave=True,
	)


def load_data(dataset_path, max_mc, flightPathLength, pulse_mode, tdc, variables, processing_mode=True,
			  load_tdc_raw=False):
	"""Load a calibration dataset and stash it on ``variables``.

	Parameters
	----------
	load_tdc_raw : bool, default False
		If True and ``tdc == 'pyccapt'``, also load the raw ``/tdc`` group from
		the h5 file. The dld and tdc dataframes are linked via a shared
		``event_group_id`` column so dld filtering decisions can later be
		propagated to tdc at save time. Stored on ``variables.data_tdc``.
	"""
	if tdc == 'pyccapt':
		# Check that the dataset is a valid pyccapt dataset with .h5 extension
		if not dataset_path.endswith(('.h5', '.H5')):
			raise ValueError('The dataset should be a valid pyccapt dataset with .h5 extension')
	elif tdc == 'leap_epos':
		# Check that the dataset is a valid leap_epos dataset with .epos extension
		if not dataset_path.endswith(('.epos', '.EPOS')):
			raise ValueError('The dataset should be a valid leap_epos dataset with .epos extension')
	elif tdc in {'pos', 'leap_pos'}:
		# Check that the dataset is a valid pos dataset with .pos extension
		if not dataset_path.endswith(('.pos', '.POS')):
			raise ValueError('The dataset should be a valid pos dataset with .pos extension')
	elif tdc == 'leap_apt':
		# Check that the dataset is a valid leap_apt dataset with .apt extension
		if not dataset_path.endswith(('.apt', '.APT')):
			raise ValueError('The dataset should be a valid leap_apt dataset with .apt extension')
	elif tdc == 'ato_v6':
		# Check that the dataset is a valid ato_v6 dataset with .ato extension
		if not dataset_path.endswith(('.ato', '.ATO')):
			raise ValueError('The dataset should be a valid ato_v6 dataset with .ato extension')

	if processing_mode:
		# Calculate the maximum possible time of flight (TOF)
		max_tof = int(tof_tools.mc2tof(max_mc, 1000, 0, 0, flightPathLength))
		print('The maximum possible TOF is:', max_tof, 'ns')
		print('=============================')
		variables.pulse_mode = pulse_mode
		dataset_path_obj = Path(dataset_path)
		variables.path = str(dataset_path_obj)
		variables.dataset_name = dataset_path_obj.stem
		output_dir = dataset_path_obj.parent / variables.dataset_name / 'data_processing'
		variables.set_result_data_directory(output_dir)
		variables.result_data_name = variables.dataset_name
		variables.set_result_directory(output_dir)

		print('The data will be saved on the path:', variables.result_data_path)
		print('=============================')
		print('The dataset name after saving is:', variables.result_data_name)
		print('=============================')
		print('The figures will be saved on the path:', variables.result_path)
		print('=============================')

		# Create data frame out of hdf5 file dataset
		tdc_df = None
		if tdc == 'pyccapt':
			try:
				loaded = data_tools.load_data(dataset_path, tdc, mode='raw', load_tdc=load_tdc_raw)
				if isinstance(loaded, tuple):
					dld_group_storage, tdc_df = loaded
				else:
					dld_group_storage = loaded
				print('The data is loaded in raw mode' + (' (with raw tdc)' if tdc_df is not None else ''))
				mode = 'raw'
			except Exception:
				loaded = data_tools.load_data(dataset_path, tdc, mode='processed', load_tdc=load_tdc_raw)
				if isinstance(loaded, tuple):
					dld_group_storage, tdc_df = loaded
				else:
					dld_group_storage = loaded
				print('The data is loaded in processed mode' + (' (with raw tdc)' if tdc_df is not None else ''))
				if 'x (nm)' not in dld_group_storage:
					mode = 'raw'
				else:
					mode = 'processed'
		else:
			if load_tdc_raw:
				print('Note: load_tdc_raw is only supported for pyccapt h5 datasets; ignoring.')
			load_data_type = 'leap_pos' if tdc in {'pos', 'leap_pos'} else tdc
			dld_group_storage = data_tools.load_data(dataset_path, load_data_type)

		if tdc == 'pyccapt' and mode == 'raw':
			data = data_tools.remove_invalid_data(dld_group_storage, max_tof)
			data = data_tools.pyccapt_raw_to_processed(data)
		else:
			data = dld_group_storage

	elif not processing_mode:
		tdc_df = None
		max_tof = int(tof_tools.mc2tof(max_mc, 1000, 0, 0, flightPathLength))
		variables.pulse_mode = pulse_mode
		dataset_path_obj = Path(dataset_path)
		variables.path = str(dataset_path_obj)
		variables.dataset_name = dataset_path_obj.stem
		output_dir = dataset_path_obj.parent / variables.dataset_name / 'visualization'
		variables.set_result_data_directory(output_dir)
		variables.result_data_name = variables.dataset_name
		variables.set_result_directory(output_dir)

		print('The data will be saved on the path:', variables.result_data_path)
		print('=============================')
		print('The dataset name after saving is:', variables.result_data_name)
		print('=============================')
		print('The figures will be saved on the path:', variables.result_path)
		print('=============================')

		# Create data frame out of hdf5 file dataset
		if tdc == 'pyccapt':
			loaded = data_tools.load_data(dataset_path, tdc, mode='processed', load_tdc=load_tdc_raw)
			if isinstance(loaded, tuple):
				data, tdc_df = loaded
			else:
				data = loaded
		else:
			if load_tdc_raw:
				print('Note: load_tdc_raw is only supported for pyccapt h5 datasets; ignoring.')
			load_data_type = 'leap_pos' if tdc in {'pos', 'leap_pos'} else tdc
			data = data_tools.load_data(dataset_path, load_data_type)

	print('Total number of Ions:', len(data))
	if tdc_df is not None:
		print('Loaded raw tdc rows:', len(tdc_df))

	variables.data = data
	variables.data_tdc = tdc_df
	variables.data_tdc_backup = tdc_df.copy() if tdc_df is not None else None
	variables.max_mc = max_mc
	variables.max_tof = max_tof
	variables.flight_path_length = flightPathLength
	variables.pulse_mode = pulse_mode
	variables.sync_from_data(update_backups=True)



def add_columns(variables, max_mc):

	if 'x (nm)' not in variables.data:
		variables.data.insert(0, 'x (nm)', np.zeros(len(variables.dld_t)))
	if 'y (nm)' not in variables.data:
		variables.data.insert(1, 'y (nm)', np.zeros(len(variables.dld_t)))
	if 'z (nm)' not in variables.data:
		variables.data.insert(2, 'z (nm)', np.zeros(len(variables.dld_t)))
	if 'mc (Da)' not in variables.data:
		variables.data.insert(4, 'mc (Da)', np.zeros(len(variables.dld_t)))
	if 'mc_uc (Da)' not in variables.data:
		variables.data.insert(5, 'mc_uc (Da)', variables.mc_uc)
	else:
		variables.data['mc_uc (Da)'] = variables.mc_uc
	if 't_c (ns)' not in variables.data:
		variables.data.insert(8, 't_c (ns)', np.zeros(len(variables.dld_t)))

	# Remove the data with mc biger than max mc
	mask = (variables.data['mc (Da)'].to_numpy() > max_mc.value)
	print('The number of data over max_mc:', len(mask[mask == True]))
	variables.data.drop(np.where(mask)[0], inplace=True)
	variables.data.reset_index(inplace=True, drop=True)

	# Remove the data with x,y,t = 0
	mask1 = (variables.data['x (nm)'].to_numpy() == 0)
	mask2 = (variables.data['y (nm)'].to_numpy() == 0)
	mask3 = (variables.data['t (ns)'].to_numpy() == 0)
	mask = np.logical_and(mask1, mask2)
	mask = np.logical_and(mask, mask3)
	print('The number of data with having t, x, and y equal to zero is:', len(mask[mask == True]))
	variables.data.drop(np.where(mask)[0], inplace=True)
	variables.data.reset_index(inplace=True, drop=True)
	variables.data_backup = variables.data.copy()


def load_calibrated_h5(dataset_path, variables, *, range_path=None, show_progress=True):
	"""Load a PyCCAPT ``.h5`` (or LEAP ``.RHIT``) file for raw-data analysis.

	Three file layouts are supported, dispatched by file extension:

	1. **Calibrated PyCCAPT bundle** (``.h5``, preferred) — produced by
	   ``data_tools.save_data(..., save_tdc=True, save_range=True)``. Contains
	   ``/df`` (calibrated dld), and optionally ``/tdc`` (raw delay-line
	   timestamps linked via ``event_group_id``) and ``/range`` (identified
	   ion windows).
	2. **Pure raw PyCCAPT acquisition** (``.h5``) — the file as written by
	   the control software: ``/dld`` and ``/tdc`` groups (no ``/df``,
	   no ``/range``). The dld records are converted to the processed
	   dataframe schema in memory, and the linked raw tdc rows are kept
	   on ``variables.data_tdc``.
	3. **LEAP CAMECA RHIT** (``.rhit``) — a Cameca-LEAP ROOT bundle. Decoded
	   via :func:`pyccapt.calibration.leap_tools.cameca_raw.rhit_load` and
	   converted to the processed dataframe schema with
	   :func:`...rhit_to_ccapt`. RHIT files have no raw delay-line tdc data
	   (the mass/charge and detector positions are already calibrated by the
	   instrument), so ``variables.data_tdc`` is set to ``None`` and the
	   downstream analyses that require ``/tdc`` (DLTS-per-pulse, combinatorial
	   recovery) are skipped automatically.

	Older datasets that store the range table in a separate
	``<dataset>_range.h5`` file are also supported via ``range_path``.

	Populates ``variables.data``, ``variables.data_tdc``, ``variables.range_data``,
	and the standard backup fields.
	"""
	import pandas as pd  # local import to keep optional Jupyter loaders fast

	from pyccapt.calibration.data_tools import data_loadcrop, data_tools

	progress = _loader_progress(total=5, enabled=show_progress, desc="Loading PyCCAPT HDF5")
	try:
		progress.set_postfix_str("validating dataset path")
		dataset_path_obj = Path(dataset_path)
		if not dataset_path_obj.is_file():
			raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

		variables.path = str(dataset_path_obj)
		variables.dataset_name = dataset_path_obj.stem
		output_dir = dataset_path_obj.parent / variables.dataset_name / 'raw_analysis'
		variables.set_result_data_directory(output_dir)
		variables.set_result_directory(output_dir)
		variables.result_data_name = variables.dataset_name
		progress.update(1)

		suffix = dataset_path_obj.suffix.lower()

		dld_df = None
		tdc_df = None
		# LEAP RHIT path: the file is a Cameca ROOT bundle, not HDF5. Trying
		# pd.read_hdf on it raises HDF5ExtError ("file signature not found")
		# because it isn't HDF5 at all — so we MUST dispatch on extension
		# before the calibrated-bundle attempt.
		if suffix == ".rhit":
			progress.set_postfix_str("reading LEAP RHIT raw data")
			from pyccapt.calibration.leap_tools.cameca_raw import (
				rhit_load,
				rhit_to_ccapt,
			)

			hits, _histograms, _metadata = rhit_load(str(dataset_path_obj))
			dld_df = rhit_to_ccapt(hits)
			tdc_df = None
			source = "leap_rhit"
			progress.update(1)   # step 2: read
			progress.update(1)   # step 3: normalize (already processed-schema)
		elif suffix in {".str", ".hits"}:
			# LEAP STR/HITS path: same story as RHIT — these are Cameca raw
			# bundles, not HDF5, so pd.read_hdf would raise HDF5ExtError.
			progress.set_postfix_str("reading LEAP STR/HITS raw data")
			from pyccapt.calibration.leap_tools.cameca_raw import (
				str_calculate_positions,
				str_load,
				str_to_ccapt,
			)

			hits, _metadata = str_load(str(dataset_path_obj))
			hits = str_calculate_positions(hits)
			dld_df = str_to_ccapt(hits)
			tdc_df = None
			source = "leap_str"
			progress.update(1)   # step 2: read
			progress.update(1)   # step 3: normalize (already processed-schema)
		else:
			# First try the calibrated bundle layout (/df). If that fails
			# because the file only has the raw acquisition groups, fall back
			# to raw /dld + /tdc, then to raw /dld only.
			source = "calibrated"
			progress.set_postfix_str("reading bundled /df and optional /tdc")
			try:
				loaded = data_tools.load_data(
					str(dataset_path_obj), 'pyccapt', mode='processed', load_tdc=True
				)
				if isinstance(loaded, tuple):
					dld_df, tdc_df = loaded
				else:
					dld_df, tdc_df = loaded, None
			except (KeyError, ValueError) as calibrated_error:
				progress.set_postfix_str("falling back to raw /dld + /tdc groups")
				try:
					dld_df, tdc_df = data_loadcrop.fetch_dataset_with_tdc(str(dataset_path_obj))
					source = "raw"
				except Exception as paired_error:
					# Bundled /tdc either is missing or its start_counter does
					# not align with /dld (e.g. our pyccapt-raw RHIT export
					# has a single-channel tdc mirror that is just redundant
					# with dld; pyccapt-raw STR has many tdc rows per event
					# and that alignment may also fail on filtered exports).
					# For the rest of the calibration workflow, /dld alone is
					# sufficient. Try dld-only before giving up.
					progress.set_postfix_str("falling back to /dld only (no tdc)")
					dld_only = data_loadcrop.fetch_dataset_from_dld_grp(
						str(dataset_path_obj), extract_mode='dld'
					)
					if dld_only is None:
						# Inspect the file to give a targeted error. The most
						# common cause is a pyccapt-raw STR export that was
						# saved before calibration was applied: that file has
						# a /tdc group (raw delay-line counts) but no /dld
						# group (which only exists once VDC / tof_ns / detx /
						# dety have been calibrated against a matching RHIT).
						import h5py
						has_dld = False
						has_tdc = False
						try:
							with h5py.File(str(dataset_path_obj), 'r') as _hf:
								top_keys = list(_hf.keys())
								has_dld = 'dld' in top_keys
								has_tdc = 'tdc' in top_keys
						except OSError:
							top_keys = []
						if has_tdc and not has_dld:
							raise ValueError(
								f"{dataset_path!r} only has a /tdc group (raw delay-line "
								"counts) — no /dld group, so it cannot be loaded into the "
								"calibration workflow. This typically happens when a STR "
								"file was exported as pyccapt-raw HDF5 *before* clicking "
								"'Calibrate from RHIT' in the cameca raw import workflow. "
								"Re-export it from cameca raw import after running calibration, "
								"or load it via the raw-data-analysis workflow which uses /tdc "
								"directly."
							) from paired_error
						raise ValueError(
							f"Could not load {dataset_path!r}: not a calibrated bundle "
							f"({calibrated_error}); /dld + /tdc paired load failed "
							f"({paired_error}); /dld-only load also failed. "
							f"File top-level groups: {top_keys}."
						) from paired_error
					dld_df = dld_only
					tdc_df = None
					source = "raw_dld_only"
					print(
						"Loaded /dld group only -- /tdc was missing or its "
						"start_counter does not align with /dld. The rest of "
						"the calibration workflow does not require /tdc; only "
						"DLTS-per-pulse / combinatorial recovery diagnostics "
						"will be unavailable."
					)
			progress.update(1)

			progress.set_postfix_str("normalizing loaded data")
			if source in ("raw", "raw_dld_only"):
				# Apply the standard raw -> processed pipeline so downstream analyses
				# see the same dataframe schema as a calibrated load.
				dld_df = data_tools.remove_invalid_data(dld_df, max_tof=100000)
				dld_df = data_tools.pyccapt_raw_to_processed(dld_df)
			progress.update(1)

		variables.data = dld_df
		variables.data_backup = dld_df.copy()
		variables.data_tdc = tdc_df
		variables.data_tdc_backup = tdc_df.copy() if tdc_df is not None else None

		# Range table: prefer /range in the same h5, then explicit range_path,
		# then fall back to <dataset>_range.h5 next to the file.
		progress.set_postfix_str("loading range table")
		range_df = None
		# Only PyCCAPT HDF5 files can carry an embedded /range group; trying
		# pd.read_hdf on a Cameca RHIT raises HDF5ExtError ("file signature
		# not found") which is NEITHER KeyError NOR ValueError, so it would
		# escape the except below. Skip the embedded read for non-HDF5 files.
		if suffix in {".h5", ".hdf5", ".hdf"}:
			try:
				range_df = pd.read_hdf(str(dataset_path_obj), key='range', mode='r')
			except (KeyError, ValueError):
				range_df = None

		if range_df is None and range_path:
			range_df = data_tools.read_range(range_path)
		if range_df is None:
			sibling = dataset_path_obj.with_name(f"{dataset_path_obj.stem}_range.h5")
			if sibling.is_file():
				range_df = data_tools.read_range(str(sibling))

		if range_df is not None:
			variables.range_data = range_df
			variables.range_data_backup = range_df.copy()
		progress.update(1)

		progress.set_postfix_str("synchronizing shared variables")
		variables.sync_from_data(update_backups=True)
		progress.update(1)
		progress.set_postfix_str("done")

		print(f"Loaded {source} dld rows: {len(dld_df)}")
		if tdc_df is not None:
			print(f"Loaded raw tdc rows:        {len(tdc_df)}")
		if range_df is not None:
			print(f"Loaded range rows:          {len(range_df)}")
		return dld_df, tdc_df, range_df
	finally:
		progress.close()
