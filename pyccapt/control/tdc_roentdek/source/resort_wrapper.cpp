//////////////////////////////////////////////////////////////////////////////////////////
//
//  roentdek_resort_x64.dll  -  RoentDek Resort64c (the reconstruction library used by
//  CoboldPC / DAn) for PyCCAPT.  Separate from the acquisition DLL on purpose: if this
//  DLL (or resort64c_VS2010_x64.dll / the VC++ 2010 runtime) is missing, the acquisition
//  still works and PyCCAPT falls back to its own reconstruction.
//
//  The sorter is configured like DAn does it from the Cobold DAn ccf parameters
//  (see roentdek_resort.cfg, made from a Cobold ccf by roentdek_offline.py ccf2resort):
//
//     ccf 1102..1104  fu, fv, fw      -> sorter->fu = 0.5 * p1102 ...
//     ccf 1105        w offset        -> shift_layer_w(+1, w_offset)
//     ccf 1106/1107   x/y offset      -> shift_position_origin(+1, x, y)
//     ccf 1108..1110  time sum offsets-> shift_sums(+1, su, sv, sw)
//     ccf 1111        MCP radius
//     ccf 1115..1117  time sum gates (half width)
//     ccf 1118..1120  runtimes (0 = same as u)
//     ccf 1121/1122   dead times
//     ccf 1123        advanced reconstruction: 1 -> sort(), 0 -> run_without_sorting()
//     ccf 1124/1125   position dependent sum / position correction (correction table)
//     ccf 1129..1135  TDC channels (Cobold numbering, 1 = first channel)
//     ccf 1137        use MCP
//     ccf 1421        start channel; TOF = particle time - start time
//
//  These conventions were verified by replaying run 198 (Jan 2024) through Resort64c and
//  comparing with the CoboldPC export of that run (same method for 98.4 % of the hits,
//  identical x/y for the all-signal hits).
//
//////////////////////////////////////////////////////////////////////////////////////////

#ifndef WIN64
#define WIN64
#endif
#ifndef WIN32
#define WIN32
#endif

#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <limits>

#include "resort64c.h"
#pragma comment(lib, "resort64c_VS2010_x64.lib")

#define NCH 8
#define NHITS 10
#define OUT_COLS 8   // event, hit, x, y, tof, method, particle time, start time

static char g_error[512] = "";

class ResortWrapper {
public:
	std::map<std::string, double> P;
	sort_class* sorter;
	__int32 cnt[NCH];
	double tdc_ns[NCH][NHITS];
	int ch_start;      // 0-based, -1 = none
	int start_selection;   // 0 = first start hit (Cobold), 1 = last start before the particle
	bool require_start;    // true: no start -> no particle (Cobold); false: particle with TOF = NaN
	bool advanced;
	double su, sv, sw, w_off, px, py, t0;
	int n_table_points;

	double p(const char* k, double d) { auto it = P.find(k); return it == P.end() ? d : it->second; }

	ResortWrapper() : sorter(NULL), ch_start(-1), start_selection(1), require_start(false), advanced(false), n_table_points(0) {}
	~ResortWrapper() { if (sorter) delete sorter; }

	bool load_config(const char* fn, std::string& table_file) {
		FILE* f = fopen(fn, "r");
		if (!f) { _snprintf(g_error, 511, "cannot open config %s", fn); return false; }
		char line[1024];
		while (fgets(line, sizeof(line), f)) {
			char* c = strchr(line, '#'); if (c) *c = 0;
			char key[256], val[768];
			if (sscanf(line, "%255s %767[^\r\n]", key, val) != 2) continue;
			if (!strcmp(key, "correction_table")) { table_file = val; while (!table_file.empty() && table_file.back() == ' ') table_file.pop_back(); continue; }
			P[key] = atof(val);
		}
		fclose(f);
		// a relative table path is relative to the folder of the config file
		if (!table_file.empty() && !(table_file.size() > 1 && (table_file[1] == ':' || table_file[0] == '\\' || table_file[0] == '/'))) {
			std::string dir(fn);
			size_t pos = dir.find_last_of("\\/");
			dir = (pos == std::string::npos) ? std::string("") : dir.substr(0, pos + 1);
			table_file = dir + table_file;
		}
		return true;
	}

	// Cobold correction table:  DAnModule,set_correction_point,sum|pos,i_det,u|v|w,x,y;
	bool load_table(const char* fn) {
		FILE* f = fopen(fn, "r");
		if (!f) { _snprintf(g_error, 511, "cannot open correction table %s", fn); return false; }
		char line[1024];
		while (fgets(line, sizeof(line), f)) {
			char* q = strstr(line, "set_correction_point,");
			if (!q) continue;
			q += strlen("set_correction_point,");
			char kind[16] = "", det[16] = "", layer[8] = ""; double x, y;
			for (char* r = q; *r; ++r) if (*r == ',' || *r == ';') *r = ' ';
			if (sscanf(q, "%15s %15s %7s %lf %lf", kind, det, layer, &x, &y) != 5) continue;
			interpolate_class* ic = NULL;
			bool sum = !strcmp(kind, "sum");
			char L = layer[0];
			if (sum) ic = (L == 'u') ? sorter->signal_corrector->sum_corrector_U : (L == 'v') ? sorter->signal_corrector->sum_corrector_V : sorter->signal_corrector->sum_corrector_W;
			else     ic = (L == 'u') ? sorter->signal_corrector->pos_corrector_U : (L == 'v') ? sorter->signal_corrector->pos_corrector_V : sorter->signal_corrector->pos_corrector_W;
			ic->set_point(x, y);
			++n_table_points;
		}
		fclose(f);
		return true;
	}

	bool init(const char* config_file) {
		std::string table_file;
		if (!load_config(config_file, table_file)) return false;
		sorter = new sort_class();
		int c1 = (int)p("ch_u1", 1) - 1, c2 = (int)p("ch_u2", 2) - 1, c3 = (int)p("ch_v1", 3) - 1, c4 = (int)p("ch_v2", 4) - 1;
		int c5 = (int)p("ch_w1", 5) - 1, c6 = (int)p("ch_w2", 6) - 1, cm = (int)p("ch_mcp", 8) - 1;
		sorter->Cu1 = c1; sorter->Cu2 = c2; sorter->Cv1 = c3; sorter->Cv2 = c4; sorter->Cw1 = c5; sorter->Cw2 = c6;
		sorter->use_MCP = p("use_mcp", 1) != 0;
		sorter->Cmcp = sorter->use_MCP ? cm : -1;
		sorter->use_HEX = p("use_hex", 1) != 0;
		sorter->common_start_mode = p("common_start", 1) != 0;
		sorter->MCP_radius = p("mcp_radius", 40);
		sorter->uncorrected_time_sum_half_width_u = p("tsum_half_width_u", 2);
		sorter->uncorrected_time_sum_half_width_v = p("tsum_half_width_v", 2);
		sorter->uncorrected_time_sum_half_width_w = p("tsum_half_width_w", 2);
		sorter->corrected_time_sum_half_width_u = p("corrected_tsum_half_width_u", sorter->uncorrected_time_sum_half_width_u);
		sorter->corrected_time_sum_half_width_v = p("corrected_tsum_half_width_v", sorter->uncorrected_time_sum_half_width_v);
		sorter->corrected_time_sum_half_width_w = p("corrected_tsum_half_width_w", sorter->uncorrected_time_sum_half_width_w);
		sorter->use_sum_correction = p("use_sum_correction", 0) != 0;
		sorter->use_pos_correction = p("use_pos_correction", 0) != 0;
		sorter->dead_time_anode = p("dead_time_anode", 25);
		sorter->dead_time_mcp = p("dead_time_mcp", 25);
		sorter->fu = 0.5 * p("fu", 0.89);
		sorter->fv = 0.5 * p("fv", 0.917);
		sorter->fw = 0.5 * p("fw", 0.887);
		double ru = p("runtime_u", 90), rv = p("runtime_v", 0), rw = p("runtime_w", 0);
		sorter->runtime_u = ru; sorter->runtime_v = (rv > 0) ? rv : ru; sorter->runtime_w = (rw > 0) ? rw : ru;
		sorter->TDC_resolution_ns = p("tdc_resolution_ns", 0.025);
		sorter->tdc_array_row_length = NHITS;
		sorter->count = cnt;
		sorter->tdc_pointer = &tdc_ns[0][0];
		advanced = p("advanced_reconstruction", 0) != 0;
		su = p("tsum_offset_u", 0); sv = p("tsum_offset_v", 0); sw = p("tsum_offset_w", 0);
		w_off = p("w_offset", 0); px = p("pos_offset_x", 0); py = p("pos_offset_y", 0);
		t0 = p("time_zero_offset", 0);
		ch_start = (int)p("ch_start", 7) - 1;
		start_selection = (int)p("start_selection", 1);
		require_start = p("require_start", 0) != 0;
		int err = sorter->init_after_setting_parameters();
		if (err) {
			char b[400] = ""; sorter->get_error_text(err, 399, b);
			_snprintf(g_error, 511, "resort init error %d: %s", err, b);
			return false;
		}
		if (!table_file.empty() && (sorter->use_sum_correction || sorter->use_pos_correction)) {
			if (!load_table(table_file.c_str())) return false;
		}
		return true;
	}

	// returns number of output rows (one per particle), or -1 if out is too small
	int process(const unsigned __int32* cnt_in, const __int32* tdc_in, int n_events, double bin_ns, double* out, int max_rows, __int64 event_offset) {
		int rows = 0;
		for (int e = 0; e < n_events; ++e) {
			const unsigned __int32* c = cnt_in + (size_t)e * NCH;
			const __int32* t = tdc_in + (size_t)e * NCH * NHITS;
			for (int ch = 0; ch < NCH; ++ch) {
				cnt[ch] = (__int32)((c[ch] > NHITS) ? NHITS : c[ch]);
				for (int h = 0; h < NHITS; ++h) tdc_ns[ch][h] = (h < cnt[ch]) ? t[ch * NHITS + h] * bin_ns : 0.0;
			}
			// start signal (Cobold: no start -> no particle)
			bool has_start = (ch_start >= 0 && ch_start < NCH && cnt[ch_start] > 0);
			if (ch_start >= 0 && !has_start && require_start) continue;
			double starts[NHITS]; int nst = has_start ? cnt[ch_start] : 0;
			for (int h = 0; h < nst; ++h) starts[h] = tdc_ns[ch_start][h];
			// the sorter must not see the start channel as detector signal: it is not one of Cu1..Cmcp

			sorter->shift_sums(+1, su, sv, sw);
			sorter->shift_layer_w(+1, w_off);
			sorter->shift_position_origin(+1, px, py);
			int n = advanced ? sorter->sort() : sorter->run_without_sorting();
			if (n <= 0) continue;
			for (int i = 0; i < n; ++i) {
				hit_class* hh = sorter->output_hit_array[i];
				double tp = hh->time;
				double st = 0.0;
				if (nst > 0) {
					st = starts[0];
					if (start_selection == 1) {   // last start before the particle
						bool found = false;
						for (int h = 0; h < nst; ++h) if (starts[h] <= tp && (!found || starts[h] > st)) { st = starts[h]; found = true; }
						if (!found) st = starts[0];
					}
				}
				if (rows >= max_rows) return -1;
				double* r = out + (size_t)rows * OUT_COLS;
				r[0] = (double)(event_offset + e); r[1] = i; r[2] = hh->x; r[3] = hh->y;
				bool no_start = (ch_start >= 0 && nst == 0);
				double nan = std::numeric_limits<double>::quiet_NaN();
				r[4] = no_start ? nan : tp - st + t0; r[5] = hh->method; r[6] = tp; r[7] = no_start ? nan : st;
				++rows;
			}
		}
		return rows;
	}
};

extern "C"
{
	__declspec(dllexport) void* rs_create(const char* config_file)
	{
		g_error[0] = 0;
		ResortWrapper* w = new ResortWrapper();
		try {
			if (!w->init(config_file)) { delete w; return NULL; }
		}
		catch (...) { _snprintf(g_error, 511, "exception in resort initialization"); delete w; return NULL; }
		return w;
	}
	__declspec(dllexport) void rs_delete(void* h) { delete (ResortWrapper*)h; }
	__declspec(dllexport) const char* rs_last_error() { return g_error; }
	__declspec(dllexport) int rs_out_cols() { return OUT_COLS; }
	__declspec(dllexport) int rs_table_points(void* h) { return ((ResortWrapper*)h)->n_table_points; }
	__declspec(dllexport) int rs_version(int* out4)
	{
		version_number_class v = sort_class::get_version_number();
		out4[0] = v.A; out4[1] = v.B; out4[2] = v.C; out4[3] = v.D;
		return 4;
	}
	// cnt[n,8] (uint32), tdc[n,8,10] (int32, TDC bins), out[max_rows, 8] doubles
	__declspec(dllexport) int rs_process(void* h, const unsigned __int32* cnt, const __int32* tdc, int n_events,
	                                     double bin_ns, double* out, int max_rows, __int64 event_offset)
	{
		try {
			return ((ResortWrapper*)h)->process(cnt, tdc, n_events, bin_ns, out, max_rows, event_offset);
		}
		catch (...) { _snprintf(g_error, 511, "exception in resort processing"); return -2; }
	}
}
